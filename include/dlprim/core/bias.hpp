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
namespace core
{    
    // functional version of backward bias
    void enqueueBackwardBiasFilter(Tensor& dy, Tensor& dw, float beta);

    ///
    /// Add bias to t over dimentsion 1: t[:,i,:,:] = b[i]
    ///
    void add_bias(Tensor &t,Tensor &b);
} // core
} // dlprim
