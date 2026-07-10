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

    void copy_strided(  Shape shape,
		tart::buffer_ptr& src, uint32_t src_offset, Shape src_strides,
		tart::buffer_ptr& dst, uint32_t dst_offset, Shape dst_strides,
		DataType dtype_src,
		DataType dtype_tgt,
		ExecutionContext const &q);
} // core
} // dlprim

