///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/softmax.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/core/loss.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {


bool SoftmaxBase::setup_kernel_params(int sm_range)
{
    if(sm_range_ == sm_range)
        return false;
    if(sm_range <= 64)
        wg_size_ = 64;
    else if(sm_range <= 128)
        wg_size_ = 128;
    else 
        wg_size_ = 256;
    items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;

    sm_range_ = sm_range;
    int mpl = wg_size_ * items_per_wi_;
    nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
    return true;
}

Softmax::~Softmax() {}
SoftmaxWithLoss::~SoftmaxWithLoss() {}

Softmax::Softmax(Context &ctx,SoftmaxConfig const &cfg) : 
    Operator(ctx),
    cfg_(cfg),
    dtype_(float_data)
{
}

SoftmaxWithLoss::SoftmaxWithLoss(Context &ctx,SoftmaxConfig const &) : 
    Operator(ctx),
    dtype_(float_data)
{
}

void Softmax::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].shape().size() == 2 || in[0].shape().size() == 3);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    out = in;
    par.clear();
    ws = 0;
}
void SoftmaxWithLoss::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].shape().size() == 2);
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in[1].shape().total_size() == in[0].shape()[0]);
    DLPRIM_CHECK(in[1].shape()[0] == in[0].shape()[0]);
    DLPRIM_CHECK(in[1].dtype() == int32_data || in[1].dtype() == float_data);
    out = {TensorSpecs(Shape(1),dtype_)};
    if(in[1].dtype() == int32_data)
        itype_ = "int";
    else
        itype_ = "float";
    par.clear();
    ws = 0;
    setup_kernel(in[0].shape()[1]);
}

void SoftmaxWithLoss::setup_kernel(int sm_range)
{
    if(!setup_kernel_params(sm_range))
        return;
	#if 1
		throw std::runtime_error("softmax_with_loss is broken right now, don't use it");
	#else
		tart::program_ptr prog_fwd = gpu::Cache::instance().get_program(ctx_.device(),"softmax_with_loss",
																"WG_SIZE",wg_size_,
																"ITEMS_PER_WI",items_per_wi_,
																"itype",itype_,
																"CALC_LOSS",1);
		kernel_ = prog_fwd->getKernel("softmax");
		tart::program_ptr prog_bwd = gpu::Cache::instance().get_program(ctx_.device(),"softmax_with_loss",
													"WG_SIZE",wg_size_,
													"ITEMS_PER_WI",items_per_wi_,
													"itype",itype_,
													"CALC_LOSS",2);

		kernel_bwd_ = prog_bwd->getKernel("softmax");
		scal_.reset(new Scal(ctx_,dtype_));
	#endif
}

void Softmax::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 2 || in[0].size() == 3);
    out = in;
    ws = 0;
}


void SoftmaxWithLoss::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==2);
    DLPRIM_CHECK(in[0].size() == 2);
    DLPRIM_CHECK(in[1].total_size() == in[0][0]);
    out = {Shape(1)};
    ws = 0;
    setup_kernel(in[0][1]);
}

void SoftmaxWithLoss::forward_gpu_loss(Tensor &input,Tensor &label, Tensor &output, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_->setArg(p++,int(in_shape[0]));
    kernel_->setArg(p++,sm_range_);
    input.set_arg(kernel_,p);
    label.set_arg(kernel_,p);
    output.set_arg(kernel_,p);

    scal_->scale(0,output,ctx);
    
    std::vector<uint32_t> gr({in_shape[0], nd_range_/wg_size_});
    std::vector<uint32_t> wg({1, wg_size_});
    kernel_->enqueue(gr, wg);
}

void SoftmaxWithLoss::backward_gpu_loss(Tensor &input,Tensor &diff, Tensor &label,Tensor &output,float factor, ExecutionContext const &ctx)
{
    Shape in_shape = input.shape();
    DLPRIM_CHECK(int(in_shape[1]) == sm_range_);
    int p=0;
    kernel_bwd_->setArg(p++,int(in_shape[0]));
    kernel_bwd_->setArg(p++,sm_range_);
    input.set_arg(kernel_bwd_,p);
    diff.set_arg(kernel_bwd_,p);
    label.set_arg(kernel_bwd_,p);
    output.set_arg(kernel_bwd_,p);
    kernel_bwd_->setArg(p++,factor);

    std::vector<uint32_t> gr({in_shape[0], nd_range_/wg_size_});
    std::vector<uint32_t> wg({1, wg_size_});
    kernel_bwd_->enqueue(gr, wg);
}

void Softmax::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2 || input[0].shape().size()==3);
    DLPRIM_CHECK(input[0].shape() == output[0].shape());
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    {
#if 0
		// because the kernel is being extremely uncooperative
#else
        core::softmax_forward(input[0],output[0],cfg_.log,ctx);
#endif
    }
}

void Softmax::backward( std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &e)
{
    if(!input.at(0).requires_gradient)
        return;
    Tensor dx = input[0].diff;
    float accum = input[0].accumulate_gradient;
    Tensor dy = output.at(0).diff;
    Tensor y  = output.at(0).data;

	{
        core::softmax_backward(dx,y,dy,cfg_.log,accum,e);
    }
}


void SoftmaxWithLoss::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==2);
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(input[1].dtype() == dtype_ || input[1].dtype() == int32_data);
    DLPRIM_CHECK(output[0].shape().total_size() == 1);
    {
        forward_gpu_loss(input[0],input[1],output[0],ctx);
    }
}

void SoftmaxWithLoss::backward( std::vector<TensorAndGradient> &input,
                        std::vector<TensorAndGradient> &output,
                        std::vector<TensorAndGradient> &,
                        Tensor &,
                        ExecutionContext const &ec)
{
    if(!input[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[1].requires_gradient == false);
    DLPRIM_CHECK(input.size()==2);
    DLPRIM_CHECK(output.size()==1); 
    float accum = input[0].accumulate_gradient;
	{
        backward_gpu_loss(input[0].data,input[0].diff,input[1].data,output[0].diff,accum,ec);
    }
}


} // namespace

