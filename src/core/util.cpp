///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <dlprim/core/util.hpp>
#include <iostream>
#include <sstream>

namespace dlprim
{
namespace core
{

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
	calcStridedTensorInvocations(const tart::device_ptr& device, const Shape& shape)
{
	std::vector<uint32_t> range;
	uint32_t dims = shape.size();
	switch(dims)
	{
		case 1: range = {shape[0]}; break;
		case 2: range = {shape[1], shape[0]}; break;
		case 3: range = {shape[2], shape[1], shape[0]}; break;
		case 4: range = {shape[3]*shape[2], shape[1], shape[0]}; break;
		case 5: range = {shape[4]*shape[3], shape[2]*shape[1], shape[0]}; break;
		case 6: range = {shape[5]*shape[4], shape[3]*shape[2], shape[1]*shape[0]}; break;
		case 7: range = {shape[6]*shape[5]*shape[4], shape[3]*shape[2], shape[1]*shape[0]}; break;
		case 8: range = {shape[7]*shape[6]*shape[5], shape[4]*shape[3]*shape[2], shape[1]*shape[0]}; break;
	default:
		throw NotImplementedError("Invalid dimentsions count for strided copy " + std::to_string(dims));
	}
	range.resize(3, 1);
	return device->chooseGlobalAndLocalSize(range);
}

void copy_strided(  Shape shape,
					tart::buffer_ptr& src, uint32_t src_offset, Shape src_strides,
					tart::buffer_ptr& dst, uint32_t dst_offset, Shape dst_strides,
					const tart::DType& dt_src,
					const tart::DType dt_dst)
{
	DLPRIM_CHECK(shape.size() == src_strides.size());
	DLPRIM_CHECK(shape.size() == dst_strides.size());
	int dims = shape.size();
	tart::device_ptr device = src->getDevice();
	bool use_io_type = dt_src == dt_dst;
	tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().copy_strided(device, dt_src, dt_dst);

	tart::kernel_ptr k = prog->getKernel("copy");
	int p=0;

	// first set shape, then srcStride, then tgtStride
	std::vector<uint32_t> shapeVec(8, 1);
	std::vector<uint32_t> srcStrideVec(8, 0);
	std::vector<uint32_t> dstStrideVec(8, 0);
	for (size_t i = 0; i < shape.size(); i += 1)
	{
		shapeVec[i] = static_cast<uint32_t>(shape[i]);
		srcStrideVec[i] = static_cast<uint32_t>(src_strides[i]);
		dstStrideVec[i] = static_cast<uint32_t>(dst_strides[i]);
		
	}
	k->setArg(p++, shapeVec);
	k->setArg(p++, srcStrideVec);
	k->setArg(p++, dstStrideVec);

	k->setArg(p++,src);
	k->setArg(p++,src_offset);
	k->setArg(p++,dst);
	k->setArg(p++,dst_offset);
	
	// Ensure GPU is properly saturated
	auto globalAndLocal = calcStridedTensorInvocations(device, shape);
	auto& local = globalAndLocal.second;
	
	std::vector<uint32_t> spec = {
		local[0], local[1], local[2], dims
	};
	k->enqueue(globalAndLocal.first, spec);
}

void copy_strided(Tensor& src, Tensor& dst)
{
	auto s = src.shape().total_size() > dst.shape().total_size() ? src.shape() : dst.shape();
	auto src_buf = src.device_buffer();
	auto src_offset = src.device_offset();
	auto src_strides = src.stride();
	
	auto dst_buf = dst.device_buffer();
	auto dst_offset = dst.device_offset();
	auto dst_strides = dst.stride();
	copy_strided(s,
		src_buf,
		src_offset,
		src_strides,
		dst_buf, dst_offset,
		dst_strides,
		src.dtype(),
		dst.dtype());
}

void broadcastTensors(std::vector<Tensor>& ts)
{
	if (ts.size() == 0) return; // safety to prevent overflow
	// Tensor with largest dims will be reference
	size_t dims = 0;
	size_t refIdx;
	for (size_t i = 0; i < ts.size(); i += 1)
	{
		Tensor& t = ts[i];
		if (t.shape().size() > dims)
		{
			dims = t.shape().size();
			refIdx = i;
		}
	}
	Tensor& ref = ts[refIdx];
	Shape refStartShape = ref.shape();
	Shape refStartStride = ref.stride();
	for (size_t i = 0; i < ts.size(); i += 1)
	{
		if (i != refIdx) broadcastTensors(ts[i], ref);
	}
	if (refStartShape != ref.shape() || refStartStride != ref.stride())
	{
		// Ref got changed, another pass is needed
		for (size_t i = 0; i < ts.size(); i += 1)
		{
			if (i != refIdx) broadcastTensors(ts[i], ref);
		}
	}
}

void matchDims(Shape& srcShape, Shape& srcStride, Shape& dstShape, Shape& dstStride)
{
	#if 1
		DLPRIM_CHECK(srcShape.size() > dstShape.size());
		
		// Ok, the below approach is not working for whatever reason.
		std::vector<size_t> dstShapeData(srcShape.size(), 1);
		std::vector<size_t> dstStrideData(srcShape.size(), 0);
		
		size_t jOffset = 0;
		for (size_t i = 0; i < srcShape.size(); i += 1)
		{
			
			for (size_t j = jOffset; j < dstShape.size(); j += 1)
			{
				if (dstShape[j] == srcShape[i])
				{
					jOffset = j + 1;
					dstShapeData[i] = dstShape[j];
					dstStrideData[i] = dstStride[j];
				}
			}
		}
		
		dstShape = Shape::from_range(dstShapeData.begin(), dstShapeData.end());
		dstStride = Shape::from_range(dstStrideData.begin(), dstStrideData.end());
	#else
		// find the closest compatible offset
		size_t dstShapeOffset = 0;
		for (size_t i = 0; i < srcShape.size(); i += 1)
		{
			bool found = false;
			for (size_t j = 0; j < dstShape.size(); j += 1)
			{
				// find the first dimension where they are compatible.
				if (dstShape[j] != 1 && dstShape[j] == srcShape[i] && j <= i)
				{
					dstShapeOffset = i - j;
					found = true;
					break;
				}
			}
			if (found) break;
		}
		std::vector<size_t> dstShapeData(srcShape.size(), 1);
		std::vector<size_t> dstStrideData(srcShape.size(), 0);
		
		for (size_t i = 0; i < dstShape.size(); i += 1)
		{
			dstShapeData[i + dstShapeOffset] = dstShape[i];
			dstStrideData[i + dstShapeOffset] = dstStride[i];
		}
		
		dstShape = Shape::from_range(dstShapeData.begin(), dstShapeData.end());
		dstStride = Shape::from_range(dstStrideData.begin(), dstStrideData.end());
	#endif
}

void broadcastTensors(Tensor& src, Tensor& dst, bool reduceDst)
{
	Shape srcShape = src.shape();
	Shape dstShape = dst.shape();
	
	if (srcShape == dstShape) return;
	
	Shape srcStride = src.stride();
	Shape dstStride = dst.stride();
	
	DLPRIM_CHECK(srcStride.size() == srcShape.size());
	DLPRIM_CHECK(dstStride.size() == dstShape.size());
	
	if (src.shape().size() > dst.shape().size())
	{
		matchDims(srcShape, srcStride, dstShape, dstStride);
	}
	else if (src.shape().size() < dst.shape().size())
	{
		// matchDims always assumes dst is the smaller one, so reverse their positions if this is not the case
		matchDims(dstShape, dstStride, srcShape, srcStride);
	}
	
	Shape srcShapeUnsqueezed = srcShape;
	Shape dstShapeUnsqueezed = dstShape;

	size_t totalDims = dstShape.size();
	// correct dimensions, set applicable strides to zero
	for (size_t i = 0; i < totalDims; i += 1)
	{
		if (srcShape[i] == 1)
		{
			srcShape[i] = dstShape[i];
			srcStride[i] = 0;
		}
		else if (dstShape[i] == 1)
		{
			if (!reduceDst)
			{
				dstShape[i] = srcShape[i];
				dstStride[i] = 0;
			}
		}
		else if (srcShape[i] != dstShape[i])
		{
			std::stringstream ss;
			ss << "Shapes are not broadcastable:\n"
				<< "	" << src.shape()
				<< "\n	" << dst.shape() 
				<< "\nUnsqueezed:\n"
				<< "	" << srcShapeUnsqueezed
				<< "\n	" << dstShapeUnsqueezed
				<< std::endl;
			throw ValidationError(ss.str());
		}
	}
	src = Tensor(src.device_buffer(), src.device_offset(), srcShape, srcStride, src.dtype());
	dst = Tensor(dst.device_buffer(), dst.device_offset(), dstShape, dstStride, dst.dtype());
	
}

} // core
} // dlprim

