///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/nll_loss.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/loss.hpp>
#include <math.h>

namespace dlprim {

NLLLoss::NLLLoss(Context &ctx,NLLLossConfig const &cfg) : 
    Operator(ctx),
    cfg_(cfg)
{
}
NLLLoss::~NLLLoss(){}
void NLLLoss::setup(std::vector<TensorSpecs> const &in,
                   std::vector<TensorSpecs> &out,
                   std::vector<TensorSpecs> &parameters,
                   size_t &workspace)
{
    DLPRIM_CHECK(in.size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in[0].shape().size()==2);
    DLPRIM_CHECK(in[1].shape().size()==1);
    DLPRIM_CHECK(in[1].shape()[0] == in[0].shape()[0]);

    size_t dim = cfg_.reduce == NLLLossConfig::reduce_none ? in[0].shape()[0] : 1;
    out = { TensorSpecs(Shape( dim ),float_data) };
    parameters.clear();
    workspace = 0;
}

void NLLLoss::reshape(std::vector<Shape> const &in,
                     std::vector<Shape> &out,
                     size_t &ws)
{
    size_t dim = cfg_.reduce == NLLLossConfig::reduce_none ? in[0][0] : 1;
    out = { Shape(dim) };
    ws = 0;
}

void NLLLoss::forward(std::vector<Tensor> &input,
                     std::vector<Tensor> &output,
                     std::vector<Tensor> &parameters,
                     Tensor &workspace)
{
    Tensor x=input.at(0);
    Tensor lbl=input.at(1);
    Tensor y=output.at(0);
    {
        float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/x.shape()[0] : 1.0f;
        core::nll_loss_forward(x,lbl,y,
                                cfg_.reduce != NLLLossConfig::reduce_none,
                                scale);
    }
}

void NLLLoss::backward(  std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &)
{
    if(!input.at(0).requires_gradient)
        return;
    Tensor dx = input[0].diff;
    Tensor lbl = input.at(1).data;
    Tensor dy = output.at(0).diff;
    float accum = input[0].accumulate_gradient;
    {
        float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/dx.shape()[0] : 1.0f;
        core::nll_loss_backward(dx,lbl,dy,
                            cfg_.reduce != NLLLossConfig::reduce_none,
                            scale,
                            accum,
                            e);
    }
}

} // dlprim
