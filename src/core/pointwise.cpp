///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
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
    
    std::vector<uint32_t> calcStridedBatchOffsets(Tensor x, size_t batchDims = 1)
    {
		if (batchDims != 1) throw std::runtime_error("not implemented");
		
		// if x is just a single - dimension tensor, it only needs 1 batch
		if (x.shape().size() == batchDims) return {x.device_offset()};
		
		std::array<size_t, max_tensor_dim> newShapeData;
		for (size_t i = 0; i < x.shape().size() - 1; i += 1)
		{
			newShapeData[i] = x.shape()[i];
		}
		Shape batchShape(x.shape().size() - 1, newShapeData);
		std::cout << "	Shape: " << x.shape() << "\n	Batch shape: " << batchShape << "\n	Strides: " << x.stride() << std::endl;
		std::vector<uint32_t> offsets(batchShape.total_size());
		for (size_t i = 0; i < batchShape.total_size(); i += 1)
		{
			Shape batchPos = flatIndexToPos(i, batchShape);
			uint32_t offset = x.device_offset();
			for (size_t j = 0; j < batchPos.size(); j += 1)
			{
				offset += (x.stride()[j]*batchPos[j]);
			}
			std::cout << "		offset at " << i << ": " << offset << std::endl;
			offsets[i] = offset;
		}
		return offsets;
	}
	
	void pointwiseOpSingleBatch(
			const tart::device_ptr& device,
			uint32_t total,
			std::vector<Tensor> xs,
			const std::vector<uint32_t>& xOffsets,
			std::vector<Tensor> ys,
			const std::vector<uint32_t>& yOffsets,
			std::vector<float> ws,
			PointwiseOp op,
			const tart::kernel_ptr& k)
	{		
		size_t p = 0;
		for (size_t i = 0; i < xs.size(); i += 1)
		{
			k->setArg(p++, xs[i].device_buffer());
			k->setArg(p++, xOffsets[i]);
			k->setArg(p++, static_cast<uint32_t>(xs[i].stride()[xs[i].stride().size() - 1]));
		}
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			std::cout << "	y[" << i << "] offset: " << yOffsets[i] << std::endl;
			k->setArg(p++, ys[i].device_buffer());
			k->setArg(p++, yOffsets[i]);
			k->setArg(p++, static_cast<uint32_t>(ys[i].stride()[ys[i].stride().size() - 1]));
		}
		k->setArg(p++, total);
		k->setArg(p++, ws);
		
		auto glPair = device->chooseGlobalAndLocalSize({total, 1, 1});
		std::vector<uint32_t> spec = {
			glPair.second[0],
			glPair.second[1],
			glPair.second[2],
			static_cast<uint32_t>(ws.size()),
			static_cast<uint32_t>(op)
		};
		k->enqueue(glPair.first, spec);
	}
	
	void pointwiseOpStrided(
			std::vector<Tensor> xs,
			std::vector<Tensor> ys,
			std::vector<float> ws,
			PointwiseOp op,
			const bool forceBatched)
	{
		// This will be a starting point for implementing strided tensor functionality.
		// A re-implementation of pointwise_operation, but supporting strided, non-contiguous tensors.
		// In addition to this, it will also use pre-defined code
		DLPRIM_CHECK(xs.size() > 0 && ys.size() > 0);
		uint32_t xTotal = xs[0].shape().total_size();
		for (const auto& x : xs)
			DLPRIM_CHECK(xTotal == x.shape().total_size());
		for (const auto& y : ys)
			DLPRIM_CHECK(xTotal == y.shape().total_size());
		
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
		
		tart::kernel_ptr k = nullptr;
		if (xs.size() == 1)
		{
			if (ys.size() == 1)
			{
				tart::program_ptr prg = gpu::PerDeviceProgramCache::instance().pointwise_unary_unary(device, xs[0].dtype(), ys[0].dtype());
				k = prg->getKernel("exec");
			}
			else
			{
				throw std::runtime_error("outputArity != 1 not implemented");
			}
		}
		else if (xs.size() == 2)
		{
			if (ys.size() == 1)
			{
				tart::program_ptr prg = gpu::PerDeviceProgramCache::instance().pointwise_binary_unary(device,
					xs[0].dtype(), xs[1].dtype(), ys[0].dtype());
				k = prg->getKernel("exec");
			}
			else
			{
				throw std::runtime_error("outputArity != 1 not implemented");
			}
		}
		
		if (allContiguous && ! forceBatched)
		{
			std::vector<uint32_t> xOffsets(xs.size());
			for (size_t i = 0; i < xs.size(); i += 1)
				xOffsets[i] = xs[i].device_offset();
			std::vector<uint32_t> yOffsets(ys.size());
			for (size_t i = 0; i < ys.size(); i += 1)
				yOffsets[i] = ys[i].device_offset();
			pointwiseOpSingleBatch(device, xTotal, xs, xOffsets, ys, yOffsets, ws, op, k);
		}
		else
		{
			// This is a lot of code. It could probably use some refactoring
			// Last dimensions need to be the same, since this determines the global size required
			DLPRIM_CHECK(xs[0].shape().size() > 0 && xs[0].shape().size() > 0);
			DLPRIM_CHECK(xs[0].shape()[xs[0].shape().size() - 1] == ys[0].shape()[ys[0].shape().size() - 1]);
		
			std::vector<std::vector<uint32_t>> xOffsetBatches(xs.size());
			for (size_t i = 0; i < xs.size(); i += 1)
				xOffsetBatches[i] = calcStridedBatchOffsets(xs[i]);
			std::vector<std::vector<uint32_t>> yOffsetBatches(ys.size());
			for (size_t i = 0; i < ys.size(); i += 1)
				yOffsetBatches[i] = calcStridedBatchOffsets(ys[i]);
			DLPRIM_CHECK(xOffsetBatches.size() == xs.size() && ys.size() == yOffsetBatches.size());
			size_t numBatches = xOffsetBatches[0].size();
			for (size_t i = 0; i < xOffsetBatches.size(); i += 1) DLPRIM_CHECK(xOffsetBatches[i].size() == numBatches);
			for (size_t i = 0; i < yOffsetBatches.size(); i += 1) DLPRIM_CHECK(yOffsetBatches[i].size() == numBatches);
			
			std::vector<uint32_t> xOffsets(xs.size());
			std::vector<uint32_t> yOffsets(ys.size());
			for (size_t batch = 0; batch < numBatches; batch += 1)
			{
				for (size_t i = 0; i < xs.size(); i += 1) xOffsets[i] = xOffsetBatches[i][batch];
				for (size_t i = 0; i < ys.size(); i += 1) yOffsets[i] = yOffsetBatches[i][batch];
				pointwiseOpSingleBatch(device, xTotal, xs, xOffsets, ys, yOffsets, ws, op, k);
			}
		}
	}
	
    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<double>  ws,
                             std::string const &code)
    {
        tart::device_ptr device = nullptr;
        if (xs.size() > 0)
			device = tensorDevice(xs[0]);
		else if (ys.size() > 0)
			device = tensorDevice(ys[0]);
		else
			throw std::runtime_error("no tensors provided!");
			
		Shape ref;
		tart::DType ref_type = tart::dtypes::float32;
		
		if(xs.empty())
		{
			ref = ys[0].shape();
			ref_type = ys[0].dtype();
		}
		else
		{
			ref = xs[0].shape();
			ref_type = xs[0].dtype();
		}
		
		#if 0
			// Just a little test
			std::cout << "	Ref: " << ref << std::endl;
			for (size_t i = 0; i < ref.total_size(); i += 1)
			{
				Shape pos = flatIndexToPos(i, ref);
				std::cout << "		Ref pos at " << i << ": " << pos << std::endl;
			}
		#endif
		
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().getPointwiseOperation(device, xs, ys, ws, code);
		
        tart::kernel_ptr k = prog->getKernel("exec");
        uint32_t total = ref.total_size();
        int p=0;
        k->setArg(p++,total);
        for(Tensor &x:xs)
            x.set_arg(k,p);
        for(Tensor &y:ys)
            y.set_arg(k,p);
        for(double w:ws)
            bind_as_dtype(k,p,w, ref_type);
            
		auto glPair = device->chooseGlobalAndLocalSize({total, 1, 1});
		k->enqueue(glPair.first, glPair.second);
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

    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double> ws,
                                        std::string const &code)
    {
        std::vector<tart::DType> dts(ws.size(),ys.at(0).dtype());
        pointwise_operation_broadcast(xs,ys,ws,dts,code);
    }

    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double> ws,
                                        const std::vector<tart::DType>& dts,
                                        std::string const &code,
                                        bool shrink_dims)
    {
        DLPRIM_CHECK(!xs.empty());
        DLPRIM_CHECK(!ys.empty());
        
        #if 0
			for (auto x: xs)
				calcStridedBatchOffsets(x);
			for (auto y: ys)
				calcStridedBatchOffsets(y);
        #endif
        
        DLPRIM_CHECK(ws.size() == dts.size());
        tart::device_ptr device = tensorDevice(xs[0]);
        
        std::vector<Shape> shapes(xs.size() + ys.size());
        for(size_t i=0;i<xs.size();i++)
            shapes[i] = xs[i].shape();
        for(size_t j=0;j<ys.size();j++)
            shapes[j+xs.size()] = ys[j].shape();

        if(shrink_dims)
            shrink_broadcast_ranges(shapes);

		//
        tart::DType target_type = ys[0].dtype();
        Shape ref = shapes[xs.size()]; // ys[0]
        for(size_t i=0;i<ys.size();i++) {
            DLPRIM_CHECK(shapes[i + xs.size()] == ref);
        }

        std::vector<Shape> strides(xs.size());
        for(size_t i=0;i<xs.size();i++) {
            strides[i] = shapes[i].broadcast_strides(ref);
        }
        
      
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().getPointwiseBroadcastOperation(
			device, xs, ys, ws, dts, code, shrink_dims);
				
        tart::kernel_ptr k = prog->getKernel("exec");
        int p=0;
        bind_shape(k,p,ref);
        for(size_t i=0;i<xs.size();i++) {
            xs[i].set_arg(k,p);
            bind_shape(k,p,strides[i]);
        }
        for(Tensor &y:ys)
            y.set_arg(k,p);
        
        for(size_t i=0;i<ws.size();i++) { 
            bind_as_dtype(k,p,ws[i], dts[i]);
        }
        std::vector<uint32_t> range = get_broadcast_ndrange(ref);
        range.resize(3, 1);
        auto glPair = device->chooseGlobalAndLocalSize(range);
		k->enqueue(glPair.first, {glPair.second[0], glPair.second[1], glPair.second[2], ref.size()});
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

    ///
    /// Perform pointwise operation with both boradcasting and reduction
    ///
    /// Calculation is performed over a shape that xs and ys tensors are boradcaasted to.
    ///
    /// For example xs have shapes: (64,10,5) and (64,10,1) and ys has shape (10,1) they all
    /// broadcast to 64,10,5 and reduction is performed over dimentsions 0 and 2
    ///
    /// All ys tensors need to have same shape and be boradcastable to total shape
    ///
    /// Optional parameters can be provided that avalible in code as w0... wN, Final ys are computed as `ys[i] = alpha[i] * reduced_result + beta[i] * ys[i]`
    ///
    class PointwiseOperationBroadcastReduceImpl : public PointwiseOperationBroadcastReduce {
    public:
        
        virtual ~PointwiseOperationBroadcastReduceImpl() {}
        ///
        /// Get size of workspace in bytes needed
        ///
        virtual size_t workspace() 
        {
            return ws_size_;
        }
        ///
        /// Perform coputations
        ///
        /// \param xs - vector of input tensor
        /// \param ys - vector of output tenors
        //  \param parameters - the weight paramerters, size should match weights_count
        /// \param alpha - scale for ys, must match size of ys
        /// \param beta - scale for summation of previous ys, must match size of ys
        ///
        ///
        virtual void enqueue(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             Tensor &workspace,
                             std::vector<double> parameters,
                             std::vector<double> alpha,
                             std::vector<double> beta)
        {
			tart::device_ptr device = tensorDevice(xs[0]);
            DLPRIM_CHECK(ws_size_ == 0 || workspace.memory_size() >= ws_size_);
            int p=0;
            
            bind_shape(kernel_,p,ref_);
            int stride_id = 0;
            for(Tensor &x:xs) {
                x.set_arg(kernel_,p);
                bind_shape(kernel_,p,strides_[stride_id++]);
            }
           
            std::vector<Tensor> temp_ys; 
            std::vector<Tensor> temp_ys_outputs;
            for(size_t i=0;i<ys.size();i++) {
                if(second_stage_stride_ == 1) {
                    Tensor &y=ys[i];
                    y.set_arg(kernel_,p);
                    bind_shape(kernel_,p,strides_[stride_id++]);
                    bind_as_dtype(kernel_,p,alpha.at(i), ys[i].dtype());
                    bind_as_dtype(kernel_,p,beta.at(i), ys[i].dtype());
                }
                else {
                    Tensor temp_y = workspace
                        .workspace_as_type(tart::dtypes::uint8)
                        .sub_tensor(ws_offsets_[i].first,Shape(ws_offsets_[i].second),tart::dtypes::uint8)
                        .workspace_as_type(ys[i].dtype());
                    temp_y.reshape(Shape(ys[i].shape().total_size(),second_stage_stride_));
                    temp_ys.push_back(temp_y);
                    Tensor temp_yout = ys[i].sub_tensor(0,Shape(ys[i].shape().total_size(),1),ys[i].dtype());
                    temp_ys_outputs.push_back(temp_yout);
                    temp_y.set_arg(kernel_,p);
                    bind_shape(kernel_,p,strides_[stride_id++]);
                }
            }
            
            for(double w: parameters) {
                bind_as_dtype(kernel_,p,w, target_type_);
            }

            if(second_stage_stride_ != 1) {
                kernel_->setArg(p++,uint32_t(second_stage_stride_));
            }
			std::vector<uint32_t> glob(range_.size());
			wg_range_.resize(3, 1);
			if (mWgSize > 0) wg_range_[0] = mWgSize; 
			for (size_t i = 0; i < glob.size(); i += 1)
				glob[i] = range_[i]/wg_range_[i];
			glob.resize(3, 1);
			#if 0 // Looks like there absolutely *has* to be a local size of 1, 1, 1. This is sub-optimal, but not worth changing for now at least
				auto glPair = device->chooseGlobalAndLocalSize(glob);
				glob = glPair.first;
				wg_range_ = glPair.second;
			#else
				wg_range_.resize(6, 1);
				wg_range_[3] = mReduceDims;
				wg_range_[4] = ref_.size();
				wg_range_[5] = mItemsPerWi;
				// just for safesies
				if (wg_range_[0] == 0) wg_range_[0] += 1;
			#endif
				
			
			if(second_stage_stride_ == 1) {
				kernel_->enqueue(glob, wg_range_);
            }
            else {
                kernel_->enqueue(glob, wg_range_);
                DLPRIM_CHECK(second_stage_->workspace() == 0);
                Tensor tmp;
                second_stage_->enqueue(temp_ys,temp_ys_outputs,tmp,{},alpha,beta);
            }
        }

        PointwiseOperationBroadcastReduceImpl(  const tart::device_ptr& device,
				std::vector<TensorSpecs> xs,
				std::vector<TensorSpecs> ys,
				int weights_count,
				const tart::DType& weights_type,
				std::string const &compute_code,
				std::string const &reduce_init,
				std::string const &reduce) :
			target_type_(weights_type),
			strides_(xs.size() + ys.size())
        {
			// This is where the lack of strides in dlprim comes back to bite us.
			// Since there are no strides stored in each tensor, any strides have to be re-computed every time this function is called.
			DLPRIM_CHECK(!xs.empty());
			DLPRIM_CHECK(!ys.empty());

			std::vector<Shape> shapes(xs.size() + ys.size());
			std::vector<Shape> strides(xs.size() + ys.size());
			for(size_t i=0;i<xs.size();i++)
			{
				shapes[i] = xs[i].shape();
				strides[i] = xs[i].stride();
			}
			for(size_t j=0;j<ys.size();j++)
			{
				shapes[j+xs.size()] = ys[j].shape();
				strides[j+xs.size()] = ys[j].stride();
			}

			shrink_broadcast_ranges(shapes);
			
			ref_ = shapes[0]; // ys[0]
			for(size_t i=1;i<shapes.size();i++) {
				ref_ = broadcast(ref_, shapes[i]);
			}
			// all yes same
			for(size_t i=xs.size()+1;i<shapes.size();i++) {
				DLPRIM_CHECK(shapes[xs.size()] == shapes[i]);
			}

			//target_type_ = weights_type;
			params_count_ = weights_count;
			ws_size_ = 0;

			for(size_t i=0;i<shapes.size();i++)
			{
				strides_[i] = shapes[i].broadcast_strides(ref_);
				#if 0
					if (strides_[i] != strides[i])
					{
						std::cout << "	shape: ";
						for(size_t j = 0; j < shapes[i].size(); j += 1)
							std::cout << shapes[i][j] << ", ";
						std::cout << "\n	broadcasted strides: ";
						for (size_t j = 0; j < strides_[i].size(); j += 1)
							std::cout << strides_[i][j] << ", ";
						std::cout << "\n	regular strides: ";
						for (size_t j = 0; j < strides[i].size(); j += 1)
							std::cout << strides[i][j] << ", ";
						std::cout << std::endl;
					}
				#endif
			}
			
			std::vector<int> reduce_dims,non_reduce_dims;
			{
				Shape ref_stride = strides_[xs.size()];
				for(int dim = 0;dim < ref_.size();dim++) {
					if(ref_stride[dim] == 0)
						reduce_dims.push_back(dim);
					else
						non_reduce_dims.push_back(dim);
				}
			}

			for(size_t i=0;i<shapes.size() + 1;i++) {
				Shape &src = i < shapes.size() ? strides_[i] : ref_;
				Shape tgt = src;
				for(int dim=0;dim<ref_.size();dim++) {
					int pos = 0;
					for(auto indx : reduce_dims)
						tgt[pos++] = src[indx];
					for(auto indx : non_reduce_dims)
						tgt[pos++] = src[indx];
				}
				src = tgt;
			}

			size_t total_reduce = 1;
			second_stage_stride_ = 1;
			for(unsigned i=0;i<reduce_dims.size();i++)
				total_reduce *= ref_[i];

			bool small_reduction = 0;
			getWorkgroupAndReductionType(small_reduction, mWgSize, total_reduce);
            
            int nd_range;
            if (!small_reduction)
            {
                mItemsPerWi = (total_reduce + mWgSize - 1) / mWgSize;
                if(mItemsPerWi >= 256) {
                    second_stage_stride_ = 256;
                }
                else if(mItemsPerWi >= 128) {
                    second_stage_stride_ = 128;
                }
                else if(mItemsPerWi >= 64) {
                    second_stage_stride_ = 64;
                }
                if(second_stage_stride_ > 1) {
                    nd_range = mWgSize * second_stage_stride_;
                    mItemsPerWi = (total_reduce + nd_range - 1) / nd_range;
                    std::vector<TensorSpecs> big_ys;
                    std::vector<TensorSpecs> small_ys;
                    std::ostringstream code;
                    Shape full_shape(ys[0].shape().total_size(),second_stage_stride_);
                    Shape red_shape(ys[0].shape().total_size(),1);
                    for(unsigned i=0;i<ys.size();i++) {
                        big_ys.push_back(TensorSpecs(full_shape,ys[i].dtype()));
                        small_ys.push_back(TensorSpecs(red_shape,ys[i].dtype()));
                        code << "y" << i <<"=x"<<i<<";\n";
                        size_t size = (big_ys.back().memory_size() + 15) / 16*16;
                        ws_offsets_.push_back(std::make_pair(ws_size_,size));
                        ws_size_ += size;
                    }
                    second_stage_.reset(new PointwiseOperationBroadcastReduceImpl(
                        device, big_ys,small_ys,0, tart::dtypes::float32,
                        code.str(),reduce_init,reduce));
					auto secondProg = gpu::PerDeviceProgramCache::instance().getPointwiseBroadcastReduceOperation(
						device, big_ys,small_ys,0, tart::dtypes::float32,
                        code.str(),reduce_init,reduce);
					mSecondStageKernel = nullptr;
                        
                }
                else {
                    int mpl = mWgSize * mItemsPerWi;
                    nd_range = (total_reduce + mpl - 1) / mpl * mWgSize;
                }
            }
            else // small_reduction == 0
            {
                mItemsPerWi = total_reduce;
                nd_range = 1; 
            }
            
//#define DEBUG_2STAGE            
#ifdef DEBUG_2STAGE
            std::cerr << "Items per thread/wg_size/nd_range:" << mItemsPerWi << "/" << mWgSize << "/" << nd_range<< std::endl;
#endif
			mReduceDims = reduce_dims.size();
			
			int zero = reduce_dims.size();
            if(zero == 0) {
                range_ = get_broadcast_ndrange(ref_);
                wg_range_.resize(range_.size(), 1);
            }
            else {
                range_ = get_broadcast_reduce_ndrange(ref_,zero,non_reduce_dims.size(),nd_range);
                wg_range_ = mWgSize == 0 ? std::vector<uint32_t>({1, 1, 1}) : std::vector<uint32_t>({mWgSize, 1, 1});
            }
			
			tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().getPointwiseBroadcastReduceOperation(
				device, xs, ys, weights_count, weights_type, compute_code, reduce_init, reduce);
			
			kernel_ = small_reduction ? kernel_ = prog->getKernel("exec_small") :
				(second_stage_stride_ > 1 ? kernel_ = prog->getKernel("exec_2_stage") : kernel_ = prog->getKernel("exec"));
            
        }
    private:
		// spec constant-compliant params
		uint32_t mReduceDims;
		uint32_t mWgSize;
		uint32_t mItemsPerWi;
		
		// original params
        size_t ws_size_;
        std::vector<Shape> strides_;
        std::vector<std::pair<size_t,size_t> > ws_offsets_;
        size_t params_count_;
        size_t second_stage_stride_;
        tart::DType target_type_;
		std::vector<uint32_t> range_,wg_range_;
        tart::kernel_ptr kernel_;
        Shape ref_;
        std::unique_ptr<PointwiseOperationBroadcastReduceImpl> second_stage_;
        tart::kernel_ptr mSecondStageKernel = nullptr;
    };


    ///
    /// Create objects:
    ///
    /// \param xs - vector of input tensor specs - such tensors are expected to be given to enqueue
    /// \param ys - vector of output tenorr specs - such tensors are expectred to be give to enqueue
    //  \param weights_count - size of parameters vector in enqueue
    /// \param weights_type - type of weights parameters as provided
    ///
    /// \param compute_code - OpenCL code to compute values. You can use x0, x1, ... xN as input values for each x for xs
    /// y0,.., yN for each output and w0,...,wN for each weight. For example "y0 = x0 + w0 * x1;"
    ///
    /// \param reduce_init - initalization of reduction variables `reduce_yN` for example "reduce_y0 = 0;" or "reduce_y0=-FLT_MAX;"
    /// \param reduce - code for sum reduction "reduce_y0 += y0" or max reduction "reduce_y0 = max(reduce_y0,y0)"
    ///
    std::unique_ptr<PointwiseOperationBroadcastReduce> PointwiseOperationBroadcastReduce::create(
                    const tart::device_ptr& device,
                    std::vector<TensorSpecs> xs,
                    std::vector<TensorSpecs> ys,
                    int weights_count,
                    const tart::DType& weights_type,
                    std::string const &compute_code,
                    std::string const &reduce_init,
                    std::string const &reduce)
    {
        std::unique_ptr<PointwiseOperationBroadcastReduce> r(new PointwiseOperationBroadcastReduceImpl(
                    device,xs,ys,
                    weights_count, weights_type,
                    compute_code,reduce_init,reduce));
        return r;
                                                                                                                
    }

    void pointwise_operation_broadcast_reduce(  std::vector<Tensor> xs,
                                                std::vector<Tensor> ys,
                                                std::vector<double>  ws,
                                                std::string const &compute,
                                                std::string const &reduce_init,
                                                std::string const &reduce)
    {
        std::vector<TensorSpecs> xspec,yspec;
        std::vector<double> alpha,beta;
        tart::device_ptr device = nullptr;
        for(auto& x:xs) {
			device = tensorDevice(x);
            xspec.push_back(x.specs());
        }
        for(auto& y:ys) {
			device = tensorDevice(y);
            yspec.push_back(y.specs());
            alpha.push_back(1.0);
            beta.push_back(0.0);
        }
        auto op = PointwiseOperationBroadcastReduce::create(device, xspec,yspec,
                            ws.size(),ys[0].dtype(),compute,reduce_init,reduce);
        Tensor workspace;
        if(op->workspace() > 0)
            workspace = Tensor(device, Shape(op->workspace()),tart::dtypes::uint8);
        op->enqueue(xs,ys,workspace,ws,alpha,beta);
    }


} // core
} // dlprim

