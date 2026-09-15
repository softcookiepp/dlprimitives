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
    /// Softmax backpropogation
    ///
    void softmax_backward(Tensor &dx,Tensor &y,Tensor &dy,bool log_softmax,float factor);
    
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_forward(Tensor &x,Tensor &label,Tensor &y,bool reduce,float scale);
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_backward(Tensor &dx,Tensor &label,Tensor &dy,bool reduce,float scale,float factor);
    
    // Re-attempt at writing a softmax kernel. Adapted from existing pointwiseOpBroadcastReduceStrided kernel, since the signature is similar enough.
    void softmaxAttempt2(Tensor x, Tensor y, std::vector<int> dims, bool useLogSoftmax);

} // core
} // dlprim
