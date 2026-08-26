///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/pointwise.hpp>
#include <dlprim/core/pointwise.hpp>
#include <cmath>
#include <my_cblas.hpp>
namespace dlprim {
    PointwiseBase::PointwiseBase(Context &ctx) : Operator(ctx) {}
    PointwiseBase::~PointwiseBase(){}
    void PointwiseBase::setup(std::vector<TensorSpecs> const &in,
                       std::vector<TensorSpecs> &out,
                       std::vector<TensorSpecs> &parameters,
                       size_t &workspace)
    {
        DLPRIM_CHECK(in.size() == 1);
        out.assign({in[0]});
        parameters.clear();
        workspace = 0;
    }
    void PointwiseBase::reshape(std::vector<Shape> const &in,
                         std::vector<Shape> &out,
                         size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        out.assign({in[0]});
        ws = 0;
    }
	void PointwiseBase::forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &)
    {
        DLPRIM_CHECK(input.size() == output.size());
        DLPRIM_CHECK(input[0].shape() == output[0].shape());
        DLPRIM_CHECK(input[0].dtype() == output[0].dtype());
        {
            forward_gpu(input[0],output[0]);
        }
    }

    void PointwiseBase::backward(std::vector<TensorAndGradient> &input,
                                std::vector<TensorAndGradient> &output,
                                std::vector<TensorAndGradient> &,
                                Tensor &)
    {
        DLPRIM_CHECK(input.size()==1);
        DLPRIM_CHECK(output.size()==1); 
        if(!input[0].requires_gradient)
            return;
        DLPRIM_CHECK(input[0].diff.shape() == output[0].diff.shape());
        DLPRIM_CHECK(input[0].diff.shape() == output[0].data.shape());
        float accum = input[0].accumulate_gradient;
        {
            backward_gpu(input[0].data,input[0].diff,output[0].data,output[0].diff,accum,q);
        }
    }

    void Threshold::forward_gpu(Tensor &x,Tensor &y)
    {
		#if 0
			core::pointwiseOpStrided();
		#else
			core::pointwise_operation({x},{y},{cfg_.threshold},"y0 = typeof_y0(cmp_gt(x0, w0) ? 1 : 0);");
		#endif
    }
    void Threshold::backward_gpu(Tensor &,Tensor &dx,Tensor &,Tensor &,float beta)
    {
        core::pointwise_operation({dx},{dx},{beta},"y0 = cmp_gt(w0, 0) ? x0 * w0 : 0;");
    }

    void Hardtanh::forward_gpu(Tensor &x,Tensor &y)
    {
        core::pointwise_operation({x},{y},{cfg_.min_val,cfg_.max_val},
                                    "y0=max(w0,min(w1,x0));");
    }
    void Hardtanh::backward_gpu(Tensor &x,Tensor &dx,Tensor &,Tensor &dy,float beta)
    {
        core::pointwise_operation({x,dy,dx},{dx},{cfg_.min_val,cfg_.max_val,beta},
                                    "y0 = (w2 != 0 ? (x2 * w2) : 0) +  ((w0 <= x0 && x0 <= w1) ? x1 : 0);");
    }

    void Abs::forward_gpu(Tensor &x,Tensor &y)
    {
        core::pointwise_operation({x},{y},{}, "y0=x0 >= 0 ? x0 : -x0;");
    }
    void Abs::backward_gpu(Tensor &x,Tensor &dx,Tensor &,Tensor &dy,float beta)
    {
        core::pointwise_operation({x,dy,dx},{dx},{beta},
                                    "y0 = (w0 != 0 ? (x2 * w0) : 0) +  ((x0 >= 0) ? x1 : -x1);");
    }


}
