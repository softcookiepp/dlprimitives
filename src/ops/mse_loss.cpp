///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/mse_loss.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>

namespace dlprim {

MSELossConfig MSELossConfig::from_json(json::value const &v)
{
    MSELossConfig cfg;
    char const *names[] = { "none", "sum", "mean" };
    cfg.reduce = utils::parse_enum(v,"reduce",names,cfg.reduce);
    return cfg;
}

MSELoss::MSELoss(Context &ctx,MSELossConfig const &cfg) : 
    Operator(ctx),
    cfg_(cfg)
{
}
MSELoss::~MSELoss(){}
void MSELoss::setup(std::vector<TensorSpecs> const &in,
                   std::vector<TensorSpecs> &out,
                   std::vector<TensorSpecs> &parameters,
                   size_t &workspace)
{
    DLPRIM_CHECK(in.size() == 2);
    DLPRIM_CHECK(in[1].shape() == in[0].shape());
    dtype_ = in[0].dtype();

    Shape dim = cfg_.reduce == MSELossConfig::reduce_none ? in[0].shape() : Shape(1);
    out = { TensorSpecs(dim, in[0].dtype()) };
    parameters.clear();
    workspace = 0;

	setup_gpu(in,out,workspace);
}

void MSELoss::setup_gpu(std::vector<TensorSpecs> in,std::vector<TensorSpecs> out,size_t &workspace)
{
    auto p = core::PointwiseOperationBroadcastReduce::create(ctx_,
            in,out,
            0,dtype_,
            "y0 = typeof_y0(x0) - typeof_y0(x1); y0 = y0*y0; ",
            "reduce_y0 = typeof_y0(0);",
            "reduce_y0 += y0;");
    fwd_ = std::move(p);
    workspace = fwd_->workspace();
}

void MSELoss::reshape(std::vector<Shape> const &in,
                     std::vector<Shape> &out,
                     size_t &ws)
{
    DLPRIM_CHECK(in.at(0) == in.at(1));
    Shape dim = cfg_.reduce == MSELossConfig::reduce_none ? in[0] : Shape(1);
    out = { dim };
    ws = 0;
	setup_gpu({TensorSpecs(in[0],dtype_),TensorSpecs(in[1],dtype_)},{TensorSpecs(dim,dtype_)},ws);
}

void MSELoss::forward(std::vector<Tensor> &input,
                     std::vector<Tensor> &output,
                     std::vector<Tensor> &parameters,
                     Tensor &workspace,
                     ExecutionContext const &q)
{
    Tensor a=input.at(0);
    Tensor b=input.at(1);
    Tensor y=output.at(0);
    {
        float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/a.shape().total_size() : 1.0f;
        fwd_->enqueue(input,output,workspace,{},{scale},{0},q);
    }
}

void MSELoss::backward(  std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &e)
{
    bool left = input.at(0).requires_gradient;
    bool right = input.at(1).requires_gradient;
    Tensor &da = input[0].diff;
    Tensor &a = input[0].data;
    Tensor &db = input[1].diff;
    Tensor &b = input[1].data;
    Tensor &dy = output.at(0).diff;
    float accum_0 = input[0].accumulate_gradient;
    float accum_1 = input[0].accumulate_gradient;
    float scale = cfg_.reduce == cfg_.reduce_mean ? 1.0f/da.shape().total_size() : 1.0f;
    {
        if(left && right) {
            core::pointwise_operation_broadcast({dy,a,b,da,db},{da,db},{scale,accum_0,accum_1},
                                      R"xxx(
                                        y0 = 2*(typeof_y0(x1) - typeof_y0(x2))*typeof_y0(x0)*typeof_y0(w0);
                                        y1 = -y0; 
                                        if(w1!=0)
                                            y0 += typeof_y0(x3) * typeof_y0(w1);
                                        if(w2!=0)
                                            y1 += typeof_y0(x4) * typeof_y0(w2);
                                        )xxx"
                                      ,e);
        }
        else {
            Tensor dx = left ? da : db;
            float factor = left ? scale : -scale;
            float accum = left ? accum_0 : accum_1;
            core::pointwise_operation_broadcast({dy,a,b,dx},{dx},{factor,accum},
                                      "y0 = 2*(typeof_y0(x1) - typeof_y0(x2))*typeof_y0(x0)*typeof_y0(w0);"
                                      "y0 = w1 != 0? y0 + typeof_y0(w1) * typeof_y0(x3) : y0;"
                                      ,e);
        }
    }
}

} // dlprim
