///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {
	
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
	calcStridedTensorInvocations(const tart::device_ptr& device, const Shape& shape);

// same as the bottom one, but with the tensor's own built-in strides and offset
void copy_strided(Tensor& src, Tensor& dst);

void copy_strided(Shape shape,
	tart::buffer_ptr& src, uint32_t src_offset, Shape src_strides,
	tart::buffer_ptr& dst, uint32_t dst_offset, Shape dst_strides,
	const tart::DType& dt_src,
	const tart::DType dt_dst);
	
Tensor broadcastTensors(Tensor& src, Tensor& dst);
	
} // core
} // dlprim

