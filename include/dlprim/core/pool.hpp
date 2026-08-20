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

    ///
    /// Compute output size of the tensor after pooling in specific dimentions.
    ///
    inline int calc_pooling_output_size(int in_size,int kernel,int pad,int stride,bool ceil_mode)
    {
        int padded_size = in_size + pad*2;
        DLPRIM_CHECK(padded_size >= kernel);
        int offset = ceil_mode ? (stride-1) : 0;
        int size = (padded_size - kernel + offset) / stride + 1;
        if((size - 1) * stride >= in_size + pad) {
            size--;
        }
        return size;
    }

    void pooling2dFwd(bool avg, std::array<uint32_t, 2> poolSize, std::array<uint32_t, 2> padSize, std::array<uint32_t, 2> strideSize, bool includePad, Tensor& in, Tensor& out);
    void pooling2dBwd(bool avg, const std::array<uint32_t, 2>& poolSize, const std::array<uint32_t, 2>& padSize, const std::array<uint32_t, 2>& strideSize, bool includePad, Tensor* x, Tensor& dx, Tensor& dy, float factor);
    void globalPoolingFwd(bool avg, Tensor& input, Tensor& output);
} // core
} //dlprim
