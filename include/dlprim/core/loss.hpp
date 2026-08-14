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
    /// Compute softmax output of x-> to y. if log_softmax true compute log of the output value
    ///
    void softmax_forward(Tensor &x,Tensor &y,bool log_softmax);
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

} // core
} // dlprim
