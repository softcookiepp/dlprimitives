///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/util.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {
    void bind_as_dtype(tart::kernel_ptr k,int &p,double value, const tart::DType& dt)
    {
       
		if      (dt == tart::dtypes::float64) k->setArg(p++, double(value));
        else if (dt == tart::dtypes::float32) k->setArg(p++, (float)(value));
        else if (dt == tart::dtypes::float16)
        {
			throw std::runtime_error("Binding as float16 not implemented yet");
			k->setArg(p++, float(value)); // half goes as float to kernel parameter
		}
        else if (dt == tart::dtypes::int64)   k->setArg(p++, int64_t(value));
        else if (dt == tart::dtypes::int32) k->setArg(p++, int(value));
        else if (dt == tart::dtypes::int16)   k->setArg(p++, int16_t(value));
        else if (dt == tart::dtypes::int8)   k->setArg(p++, int8_t(value));
        else if (dt == tart::dtypes::uint64)   k->setArg(p++, uint64_t(value));
        else if (dt == tart::dtypes::uint32) k->setArg(p++, uint32_t(value));
        else if (dt == tart::dtypes::uint16) k->setArg(p++, uint16_t(value));
        else if (dt == tart::dtypes::uint8)   k->setArg(p++, uint8_t(value));
		else throw std::runtime_error("Unsupported type");
    }
    
    Shape flatIndexToPos(size_t idx, const Shape& shape)
    {
		Shape pos = shape;
		size_t coef = 1;
		for (int i = shape.size() - 1; i >= 0; i -= 1)
		{
			size_t dLen = shape[i];
			size_t mod = (idx/coef) % dLen;
			pos[i] = mod;
			coef *= dLen;
		}
		return pos;
	}
	
	// max supported dims is 8, at least for now.
	struct CLShape
	{
		uint32_t s[8];
	};

    template<int size>
    void bind_cl_shape(tart::kernel_ptr k,int &p,Shape const &s)
    {
		CLShape cl_s;
        for(int i=0;i<size;i++)
            cl_s.s[i] = s[i];
        k->setArg(p++, cl_s);
    }
    void bind_shape(tart::kernel_ptr k, int &p,Shape const &s)
    {
        switch(s.size()) {
        case 1: bind_cl_shape<1>(k,p,s); return;
        case 2: bind_cl_shape<2>(k,p,s); return;
        case 3: bind_cl_shape<3>(k,p,s); return;
        case 4: bind_cl_shape<4>(k,p,s); return;
        case 5: bind_cl_shape<5>(k,p,s); return;
        case 6: bind_cl_shape<6>(k,p,s); return;
        case 7: bind_cl_shape<7>(k,p,s); return;
        case 8: bind_cl_shape<8>(k,p,s); return;
        default:
            {
                std::ostringstream ss;
                ss << "Shape isn't valid " << s;
                throw ValidationError(ss.str());
            }
        }
    }
    
    void bindShape(tart::UniformBlock& block, int& p, const Shape& s)
    {
		CLShape cl_s;
        for(int i = 0; i < s.size(); i += 1)
            cl_s.s[i] = s[i];
        block.setMemberData(p++, cl_s);
	}
	
	void pointwiseOpStrided(
			std::vector<Tensor> xs,
			std::vector<Tensor> ys,
			std::vector<float> ws,
			PointwiseOp op,
			const tart::DType& acctype,
			const tart::DType& iacctype)
	{
		// This will be a starting point for implementing strided tensor functionality.
		// A re-implementation of pointwise_operation, but supporting strided, non-contiguous tensors.
		// In addition to this, it will also use pre-defined code
		DLPRIM_CHECK(xs.size() > 0 && ys.size() > 0);
		uint32_t dims = xs[0].shape().size();
		for (const auto& x : xs)
			DLPRIM_CHECK(dims == x.shape().size());
		for (const auto& y : ys)
			DLPRIM_CHECK(dims == y.shape().size());
		
		tart::device_ptr device = tensorDevice(xs[0]);
		bool allContiguous = true;
		for (auto& x : xs)
		{
			if (!x.isContiguous())
			{
				allContiguous = false;
				break;
			}
		}
		if (allContiguous)
		{
			for (auto& y : ys)
			{
				if (!y.isContiguous())
				{
					allContiguous = false;
					break;
				}
			}
		}
		
		tart::program_ptr prg = nullptr;
		tart::kernel_ptr k = nullptr;
		if (xs.size() == 1)
		{
			if (ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_unary_unary(device, xs[0].dtype(), ys[0].dtype());
			}
			else
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_unary_binary(device,
					xs[0].dtype(), ys[0].dtype(), ys[1].dtype());
			}
		}
		else if (xs.size() == 2)
		{
			if (ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_binary_unary(device,
					xs[0].dtype(), xs[1].dtype(), ys[0].dtype());
			}
			else
			{
				throw std::runtime_error("outputArity != 1 not implemented");
			}
		}
		else if (xs.size() == 3)
		{
			if (ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_trinary_unary(device,
					xs[0].dtype(), xs[1].dtype(), xs[2].dtype(), ys[0].dtype());
			}
			else
			{
				throw std::runtime_error("outputArity != 1 not implemented");
			}
		}
		k = prg->getKernel("main");
		int p = 0;
		for (size_t i = 0; i < xs.size(); i += 1)
		{
			k->setArg(p++, xs[i].device_buffer());
			k->setArg(p++, xs[i].device_offset());
			bind_shape(k, p, xs[i].stride());
		}
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			k->setArg(p++, ys[i].device_buffer());
			k->setArg(p++, ys[i].device_offset());
			bind_shape(k, p, ys[i].stride());
		}
		Shape ref = ys[0].shape();
		bind_shape(k, p, ref);
		k->setArg(p++, ws);
		
		auto glPair = calcStridedTensorInvocations(device, ref);
		std::vector<uint32_t> spec = {
			glPair.second[0],
			glPair.second[1],
			glPair.second[2],
			static_cast<uint32_t>(ws.size()),
			static_cast<uint32_t>(op),
			static_cast<uint32_t>(ref.size())
		};
		k->enqueue(glPair.first, spec);
	}
	
	// Just like the above, but with auto-broadcasting.
	void pointwiseOpBroadcastStrided(std::vector<Tensor> xs,
		std::vector<Tensor> ys,
		std::vector<float> ws,
		const PointwiseOp op,
		const tart::DType& acctype,
		const tart::DType& iacctype)
	{
		DLPRIM_CHECK(xs.size() > 0 && ys.size() > 0);
		std::vector<Tensor> broadcasted(xs.size() + ys.size());
		for (size_t i = 0; i < xs.size(); i += 1) broadcasted[i] = xs[i];
		for (size_t i = 0; i < ys.size(); i += 1) broadcasted[i + xs.size()] = ys[i];
		broadcastTensors(broadcasted);
		for (size_t i = 0; i < xs.size(); i += 1) xs[i] = broadcasted[i];
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			Tensor& y = broadcasted[i + xs.size()];
			if (ys[i].shape().total_size() < y.shape().total_size())
				throw ValidationError("Output tensor requires reduce, cannot use");
			ys[i] = y;
		}
		pointwiseOpStrided(xs, ys, ws, op);
	}

    std::string format_code(std::string const &code)
    {
        std::ostringstream code_fixed;
        for(size_t i=0;i<code.size();i++)
            if(code[i]=='\n')
                code_fixed << "\\\n";
            else
                code_fixed << code[i];
        code_fixed << '\n';
        return code_fixed.str();
    }

    std::vector<uint32_t> get_broadcast_ndrange(Shape ref)
    {
		std::vector<uint32_t> range;
        switch(ref.size()) {
        case 1: range = {ref[0]}; break;
        case 2: range = {ref[1],ref[0]}; break;
        case 3: range = {ref[2],ref[1],ref[0]}; break;
        case 4: range = {ref[3]*ref[2],ref[1],ref[0]}; break;
        case 5: range = {ref[4]*ref[3],ref[2]*ref[1],ref[0]}; break;
        case 6: range = {ref[5]*ref[4],ref[3]*ref[2],ref[1]*ref[0]}; break;
        case 7: range = {ref[6]*ref[5]*ref[4],ref[3]*ref[2],ref[1]*ref[0]}; break;
        case 8: range = {ref[7]*ref[6]*ref[5],ref[4]*ref[3]*ref[2],ref[1]*ref[0]}; break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }
        return range;
    }
    std::vector<uint32_t> get_broadcast_reduce_ndrange(Shape ref,int zero,int non_reduce_dims,size_t nd_range)
    {
		std::vector<uint32_t> range;
        switch(non_reduce_dims) {
        case 0: range = {nd_range,1,                       1                      }; break;
        case 1: range = {nd_range,ref[zero+0],             1                      }; break;
        case 2: range = {nd_range,ref[zero+1],             ref[zero+0]            }; break;
        case 3: range = {nd_range,ref[zero+2],             ref[zero+1]*ref[zero+0]}; break;
        case 4: range = {nd_range,ref[zero+3]*ref[zero+2], ref[zero+1]*ref[zero+0]}; break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }
        return range;
    }
	
	void getWorkgroupAndReductionType(bool& smallReduction, uint32_t& wgSize, size_t total_reduce)
    {
		smallReduction = 0;
		if(total_reduce >= 256) {
			wgSize = 256;
		}
		else if(total_reduce >= 128) {
			wgSize = 128;
		}
		else if(total_reduce >= 64) {
			wgSize = 64;
		}
		else {
			wgSize = 0;
			smallReduction = 1;
		}
	}
	
	std::vector<int> getReduceDims(dlprim::Shape ref, std::vector<int> dim)
    {
		// get all the dimensions
		if (dim.empty())
		{
			dim.resize(ref.size());
			for (int i = 0; i < ref.size(); i += 1) dim[i] = i;
		}
		// check for negatives
		for (int i = 0; i < dim.size(); i += 1) dim[i] = (dim[i] >= 0) ? dim[i] : static_cast<int>(ref.size()) + dim[i];
		return dim;
	}
	
	// this one takes properly-broadcasted x an y shapes, and returns the correct axis to reduce across
	std::vector<int> getReduceDims(dlprim::Shape& x, dlprim::Shape& y)
    {
		std::vector<int> reduceDims;
		for (size_t d = 0; d < x.size(); d += 1)
		{
			if (x[d] != y[d] && y[d] == 1)
				reduceDims.push_back(static_cast<int>(d));
		}
		return reduceDims;
	}
	
	void pointwiseOpBroadcastReduceStrided(std::vector<Tensor> xs, std::vector<Tensor> ys, std::vector<float> ws,
		std::vector<int> reduceDims,
		PointwiseOp calcOp, PointwiseOp reduceOp, std::vector<float> yInitValues)
	{
		DLPRIM_CHECK(xs.size() > 0 && ys.size() > 0);
		DLPRIM_CHECK(yInitValues.size() == ys.size());
		
		tart::device_ptr device = tensorDevice(xs[0]);
		
		// same as pointwiseOpBroadcastStrided, but with a couple differences
		std::vector<Tensor> broadcasted(xs.size() + ys.size());
		for (size_t i = 0; i < xs.size(); i += 1) broadcasted[i] = xs[i];
		
		// still need ys in the broadcast to make sure xs are unsqueezed correctly, even if the resulting tensors are not used
		for (size_t i = 0; i < ys.size(); i += 1) broadcasted[i + xs.size()] = ys[i];
		broadcastTensors(broadcasted);
		
		// load xs back in
		Shape xShape = broadcasted[0].shape();
		for (size_t i = 0; i < xs.size(); i += 1)
		{
			xs[i] = broadcasted[i];
			DLPRIM_CHECK(xs[i].shape() == xShape);
		}
		
		// now broadcast ys
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			broadcastTensors(xs[0], ys[i], true);
		}
		
		Tensor y0 = ys[0];
		Shape yShape = y0.shape();
		if (reduceDims.size() == 0)
			reduceDims = getReduceDims(xShape, yShape);
		else 
			reduceDims = getReduceDims(xShape, reduceDims);
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			DLPRIM_CHECK(y0.shape() == ys[i].shape());
			for (size_t j = 0; j < reduceDims.size(); j += 1)
			{
				if (ys[i].shape()[reduceDims[j]] != 1)
				{
					// Ensure y dimensions at reduce dims are all 1.
					// Otherwise it cannot be reduced!
					std::stringstream ss;
					ss << "failed to apply reduction:"
						<< "\nreduced x: " << xs[i].shape()
						<< "\nreduced y: " << ys[i].shape()
						<< std::endl; 
					throw std::runtime_error(ss.str());
				}
			}
		}
		
		// convert it to shape so that it can be bound
		Shape reduceDimShape = Shape::from_range(reduceDims.begin(), reduceDims.end());
		Shape reduceShape = reduceDimShape;
		uint32_t numReduceElems = 1;
		for (size_t i = 0; i < reduceDimShape.size(); i += 1)
		{
			reduceShape[i] = xShape[reduceDimShape[i]];
			numReduceElems *= reduceShape[i];
		}
		#if 1
			std::vector<tart::DType> xts(xs.size(), tart::dtypes::float32);
			std::vector<tart::DType> yts(ys.size(), tart::dtypes::float32);
			for (size_t i = 0; i < xs.size(); i += 1)
				xts[i] = xs[i].dtype();
			for (size_t i = 0; i < ys.size(); i += 1)
				yts[i] = ys[i].dtype();
			tart::kernel_ptr k = gpu::PerDeviceProgramCache::instance().pointwiseReduce(device, xts, yts);
			
			// unused params still need to be provided due to layout nonsense in order to avoid pipeline bloat
			tart::UniformBlock block;
			int p = 0;
			for (size_t i = 0; i < gpu::kPointwiseMaxArityX; i += 1)
			{
				if (i < xs.size())
				{
					k->setArg(p++, xs[i].device_buffer());
					k->setArg(p++, xs[i].device_offset());
					for (size_t j = 0; j < max_tensor_dim; j += 1)
						block.setMemberData(p++, static_cast<uint32_t>(xs[i].stride()[j]));
				}
				else // just pad with first
				{
					k->setArg(p++, xs[0].device_buffer());
					k->setArg(p++, xs[0].device_offset());
					for (size_t j = 0; j < max_tensor_dim; j += 1)
						block.setMemberData(p++, static_cast<uint32_t>(xs[0].stride()[j]));
				}
			}
			
			for (size_t i = 0; i < gpu::kPointwiseMaxArityY; i += 1)
			{
				if (i < ys.size())
				{
					k->setArg(p++, ys[i].device_buffer());
					k->setArg(p++, ys[i].device_offset());
					for (size_t j = 0; j < max_tensor_dim; j += 1)
						block.setMemberData(p++, static_cast<uint32_t>(ys[i].stride()[j]));
				}
				else // just pad with first
				{
					k->setArg(p++, ys[0].device_buffer());
					k->setArg(p++, ys[0].device_offset());
					for (size_t j = 0; j < max_tensor_dim; j += 1)
						block.setMemberData(p++, static_cast<uint32_t>(ys[0].stride()[j]));
				}
			}
			for (size_t j = 0; j < max_tensor_dim; j += 1)
				block.setMemberData(p++, static_cast<uint32_t>(xShape[j]));
			for (size_t j = 0; j < max_tensor_dim; j += 1)
				block.setMemberData(p++, static_cast<uint32_t>(reduceDimShape[j]));
			k->setArg(p++, yInitValues);
			k->setArg(p++, ws);
			k->setArg(p++, 0, gpu::kPointwiseMaxArityY + gpu::kPointwiseMaxArityX, block); // set 0
			
		#else
			// Single-stage reduction.
			tart::program_ptr prg = nullptr;
			tart::kernel_ptr k = nullptr;
			if (xs.size() == 1 && ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_reduce_unary_unary(device, xs[0].dtype(), ys[0].dtype());
			}
			else if(xs.size() == 2 && ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_reduce_binary_unary(device, xs[0].dtype(), xs[1].dtype(), ys[0].dtype());
			}
			else if(xs.size() == 3 && ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_reduce_trinary_unary(
					device, xs[0].dtype(), xs[1].dtype(), xs[2].dtype(), ys[0].dtype());
			}
			else if(xs.size() == 4 && ys.size() == 1)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_reduce_quaternary_unary(
					device, xs[0].dtype(), xs[1].dtype(), xs[2].dtype(), xs[3].dtype(), ys[0].dtype());
			}
			else if(xs.size() == 1 && ys.size() == 2)
			{
				prg = gpu::PerDeviceProgramCache::instance().pointwise_reduce_unary_binary(device, xs[0].dtype(), ys[0].dtype(), ys[1].dtype());
			}
			k = prg->getKernel("main");
			if (!k) throw std::runtime_error("suitable kernel not found");
			
			int p = 0;
			for (size_t i = 0; i < xs.size(); i += 1)
			{
				k->setArg(p++, xs[i].device_buffer());
				k->setArg(p++, static_cast<uint32_t>(xs[i].device_offset()));
				bind_shape(k, p, xs[i].stride());
			}
			for (size_t i = 0; i < ys.size(); i += 1)
			{
				k->setArg(p++, ys[i].device_buffer());
				k->setArg(p++, static_cast<uint32_t>(ys[i].device_offset()));
				bind_shape(k, p, ys[i].stride());
			}
			bind_shape(k, p, xShape);
			bind_shape(k, p, reduceDimShape);
			k->setArg(p++, yInitValues);
			k->setArg(p++, ws);
		#endif
		
		
		// calculate local size and work per thread, based on the max amount of local invocations along the X axis for this device
		uint32_t wpt = 1;
		uint32_t wgxSize = numReduceElems;
		uint32_t maxWgxSize = device->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
		while (wgxSize > maxWgxSize)
		{
			wpt += 1;
			wgxSize = numReduceElems / wpt;
			if (wgxSize == 0 || numReduceElems % wpt > 0) wgxSize += 1;
		}
		// need to ensure this is invoked at all
		if (wgxSize == 0 || numReduceElems % wpt > 0) wgxSize += 1;
		
		uint32_t localMemSize = wgxSize;
		#if 0
			// this is supposed to reduce the amount of local memory required, but for some reason its is causes the kernel to compute nan.
			// Still need to figure out why.
			if (device->getMetadata().subgroupAdd)
			{
				// Less local memory is required if subgroup arithmetic reduction is used
				uint32_t subgroupSize = device->getMetadata().maxSubgroupSize;
				localMemSize = localMemSize / subgroupSize;
				if (localMemSize == 0 || localMemSize % subgroupSize > 0) localMemSize += 1;
			}
		#endif
		
		std::vector<uint32_t> global = calcStridedTensorRange(device, y0.shape());
		auto glPair = calcStridedTensorInvocations(device, y0.shape());
		#if 1
			std::vector<uint32_t> spec(11);
		#else
			std::vector<uint32_t> spec(9);
		#endif
		spec[0] = wgxSize;
		spec[1] = ws.size();
		spec[2] = static_cast<uint32_t>(xShape.size());
		spec[3] = static_cast<uint32_t>(reduceDimShape.size());
		spec[4] = numReduceElems;
		spec[5] = wpt;
		spec[6] = localMemSize;
		spec[7] = static_cast<uint32_t>(calcOp);
		spec[8] = static_cast<uint32_t>(reduceOp);
		#if 1
			spec[9] = static_cast<uint32_t>(xs.size());
			spec[10] = static_cast<uint32_t>(ys.size());
		#endif
		k->enqueue(global, spec);
	}

} // core
} // dlprim

