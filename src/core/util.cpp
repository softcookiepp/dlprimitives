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

Tensor broadcastTensors(Tensor& src, Tensor& dst)
{
	if (src.shape().size() != dst.shape().size())
	{
		throw NotImplementedError("Can't broadcast shapes of different dimensions quite yet!");
	}

	size_t totalDims = dst.shape().size();
	Shape broadcastShape = dst.shape();
	Shape newSrcStrides = src.stride();
	
	// check that the dimensions are alright, set any applicable strides to zero
	for (size_t i = 0; i < totalDims; i += 1)
	{
		DLPRIM_CHECK(src.shape()[i] == dst.shape()[i] || src.shape()[i] == 1);
		if (src.shape()[i] == 1) newSrcStrides[i] = 0;
	}
	return Tensor(src.device_buffer(), src.device_offset(), broadcastShape, newSrcStrides, src.dtype());
}

} // core
} // dlprim

