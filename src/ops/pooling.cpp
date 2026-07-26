///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/pooling.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <dlprim/ops/scal.hpp>
#include <dlprim/core/pool.hpp>
#include <my_cblas.hpp>

namespace dlprim {
PoolingBase PoolingBase::from_json(json::value const &v)
{
    PoolingBase cfg;
    char const *names[] = { "max", "avg" };
    cfg.mode = utils::parse_enum(v,"mode",names,cfg.mode);
    return cfg;
}

Pooling2DConfig Pooling2DConfig::from_json(json::value const &v)
{
    Pooling2DConfig cfg;
    static_cast<PoolingBase &>(cfg) = PoolingBase::from_json(v);
    utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
    utils::get_1dNd_from_json(v,"stride",cfg.stride);
    utils::get_1dNd_from_json(v,"pad",cfg.pad);
    cfg.ceil_mode = v.get("ceil_mode",cfg.ceil_mode);
    cfg.count_include_pad = v.get("count_include_pad",cfg.count_include_pad);
    return cfg;
}


Pooling2D::Pooling2D(Context &ctx,Pooling2DConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
    DLPRIM_CHECK(config_.kernel[0] > 0 && config_.kernel[1] > 0);
    DLPRIM_CHECK(config_.stride[0] > 0 && config_.stride[1] > 0);
    DLPRIM_CHECK(config_.pad[0] >= 0 && config_.pad[1] >= 0);
    DLPRIM_CHECK(config_.max <= config_.mode  && config_.mode <= config_.avg);
}

Pooling2D::~Pooling2D()
{
}

void Pooling2D::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    Shape ins = in[0].shape();
    Shape outs = calc_shape(ins);
    out.assign({TensorSpecs(outs,dtype_)});
    p.clear();
    ws = 0;

    if(config_.mode == PoolingBase::max) {
        fwd_ = std::move(core::Pooling2DForward::create_max_pooling(
                ctx_,
                config_.kernel,config_.pad,config_.stride,
                dtype_));
        bwd_ = std::move(core::MaxPooling2DBackward::create(
                ctx_,
                config_.kernel,config_.pad,config_.stride,
                dtype_));
    }
    else {
        fwd_ = std::move(core::Pooling2DForward::create_avg_pooling(
                ctx_,
                config_.kernel,config_.pad,config_.stride,config_.count_include_pad,
                dtype_));
        bwd_ = std::move(core::AvgPooling2DBackward::create(
                ctx_,
                config_.kernel,config_.pad,config_.stride,config_.count_include_pad,
                dtype_));
    }
    ws =std::max(fwd_->workspace(),bwd_->workspace());
}

int Pooling2D::calc_output_size(int in_size,int dim)
{
    return core::calc_pooling_output_size(in_size,  config_.kernel[dim],
                                                    config_.pad[dim],
                                                    config_.stride[dim],
                                                    config_.ceil_mode);
}

Shape Pooling2D::calc_shape(Shape ins)
{
    DLPRIM_CHECK(ins.size()==4);
    int oh = calc_output_size(ins[2],0);
    int ow = calc_output_size(ins[3],1);
    return Shape(ins[0],ins[1],oh,ow);
}

void Pooling2D::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape ins = in[0];
    out.assign({calc_shape(ins)});
        ws = std::max(fwd_->workspace(),bwd_->workspace());
}

template<typename Dtype>
struct Pooling2D::MaxRedcue {
    static constexpr Dtype init_val = -std::numeric_limits<Dtype>::max();
    static Dtype apply(Dtype a,Dtype b) { return std::max(a,b); };
    static Dtype norm_valid(Dtype a,int ,int,int,int ) { return a; }
    static Dtype norm_full(Dtype a) { return a; }
};

template<typename Dtype>
struct Pooling2D::AveReduceValid
{
    AveReduceValid(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    static Dtype norm_valid(Dtype a,int  dr,int dc,int,int) { return a * (Dtype(1)/(dr*dc)); }
    Dtype norm_full(Dtype a) { return a * factor; }
};

template<typename Dtype>
struct Pooling2D::AveReduceFull
{
    AveReduceFull(Dtype f) : factor(f) {}
    Dtype factor;
    static constexpr Dtype init_val = Dtype();
    static Dtype apply(Dtype a,Dtype b) { return a+b; };
    Dtype norm_valid(Dtype a,int,int,int dr_p,int dc_p) { return a / (dr_p*dc_p); }
    Dtype norm_full(Dtype a) { return a * factor; }
};


void Pooling2D::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(output[0].shape() == calc_shape(input[0].shape()));
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);

        forward_gpu(input[0],output[0],e);
}

void Pooling2D::backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ex)
{
    bwd_->enqueue(x,dx,dy,factor,ex);
}

void Pooling2D::forward_gpu(Tensor &in,Tensor &out,ExecutionContext const &ctx)
{
    fwd_->enqueue(in,out,ctx);
}

void Pooling2D::backward(std::vector<TensorAndGradient> &input,
                         std::vector<TensorAndGradient> &output,
                         std::vector<TensorAndGradient> &/*parameters*/,
                         Tensor &/*workspace*/,
                         ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!input[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[0].data.shape().size()==4);
    DLPRIM_CHECK(input[0].diff.shape().size()==4);
    DLPRIM_CHECK(output[0].data.shape().size()==4);

    DLPRIM_CHECK(input[0].data.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].data.shape()[1] == output[0].diff.shape()[1]);
    
    DLPRIM_CHECK(input[0].diff.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].diff.shape()[1] == output[0].diff.shape()[1]);

    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].data.dtype() == dtype_);
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
	{
        backward_gpu(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient,ctx);
    }

}





GlobalPooling::GlobalPooling(Context &ctx,GlobalPoolingConfig const &cfg) :
    Operator(ctx),
    cfg_(cfg),
    dtype_(float_data)
{
}
GlobalPooling::~GlobalPooling() {}

void GlobalPooling::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &par,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0].shape(); 
    DLPRIM_CHECK(in[0].dtype() == float_data);
    DLPRIM_CHECK(in_shape.size() == 4);
    out.assign({TensorSpecs(Shape(in_shape[0],in_shape[1],1,1),in[0].dtype())});
    par.clear();
    ws = 0;
    ws = setup_kernel(in_shape);
}

size_t GlobalPooling::setup_kernel(Shape const &in_shape)
{
    if(cfg_.mode == PoolingBase::max) {
        fwd_ = std::move(core::Pooling2DForward::create_global_max_pooling(ctx_,in_shape,dtype_));
        bwd_ = std::move(core::MaxPooling2DBackward::create_global(ctx_,in_shape,dtype_));
    }
    else {
        fwd_ = std::move(core::Pooling2DForward::create_global_avg_pooling(ctx_,in_shape,dtype_));
        bwd_ = std::move(core::AvgPooling2DBackward::create_global(ctx_,in_shape,dtype_));
    }
    return std::max(fwd_->workspace(),bwd_->workspace());
}


void GlobalPooling::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].size() == 4);
    ws = 0;
    out.assign({Shape(in[0][0],in[0][1],1,1)});
    ws=setup_kernel(in[0]);
}

void GlobalPooling::backward_gpu(Tensor &x,Tensor &dx,Tensor &dy,float factor,ExecutionContext const &ctx)
{
    bwd_->enqueue(x,dx,dy,factor,ctx);
}


void GlobalPooling::forward_gpu(Tensor &input, Tensor &output, ExecutionContext const &ctx)
{
    fwd_->enqueue(input,output,ctx);
}


void GlobalPooling::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &, Tensor &,ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(input[0].shape().size()==4);
    DLPRIM_CHECK(output[0].shape().size()==4);
    DLPRIM_CHECK(input[0].shape()[0] == output[0].shape()[0]);
    DLPRIM_CHECK(input[0].shape()[1] == output[0].shape()[1]);
    DLPRIM_CHECK(1 == output[0].shape()[2]);
    DLPRIM_CHECK(1 == output[0].shape()[3]);
    DLPRIM_CHECK(output[0].dtype() == dtype_);
    {
        forward_gpu(input[0],output[0],ctx);
    }
}

void  GlobalPooling::backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &,
                              Tensor &,
                              ExecutionContext const &ctx)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!output[0].requires_gradient)
        return;
    DLPRIM_CHECK(input[0].data.shape().size()==4);
    DLPRIM_CHECK(output[0].diff.shape().size()==4);
    DLPRIM_CHECK(input[0].diff.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].diff.shape()[1] == output[0].diff.shape()[1]);
    DLPRIM_CHECK(input[0].data.shape()[0] == output[0].diff.shape()[0]);
    DLPRIM_CHECK(input[0].data.shape()[1] == output[0].diff.shape()[1]);
    DLPRIM_CHECK(1 == output[0].diff.shape()[2]);
    DLPRIM_CHECK(1 == output[0].diff.shape()[3]);
    DLPRIM_CHECK(output[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].diff.dtype() == dtype_);
    DLPRIM_CHECK(input[0].data.dtype() == dtype_);
    {
        backward_gpu(input[0].data,input[0].diff,output[0].diff,input[0].accumulate_gradient,ctx);
    }
}



} // dlprim


