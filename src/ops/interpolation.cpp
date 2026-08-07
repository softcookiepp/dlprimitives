///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/interpolation.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {
InterpolationConfig InterpolationConfig::from_json(json::value const &v)
{
    InterpolationConfig cfg;
    cfg.out_h = v.get<int>("out_h",-1);
    cfg.out_w = v.get<int>("out_w",-1);
    cfg.scale_y = v.get<double>("scale_y",-1.0);
    cfg.scale_x = v.get<double>("scale_x",-1.0);
    cfg.align_corners = v.get<bool>("align_corners",false);
    std::string method = v.get<std::string>("method");
    if(method == "nearest")
        cfg.method = InterpolateType::nearest;
    else if(method == "nearest-exact")
        cfg.method = InterpolateType::nearest_exact;
    else if(method == "bilinear")
        cfg.method = InterpolateType::bilinear;
    else
       throw ValidationError("Unsupported interpolation method " + method); 
    return cfg;
}

Interpolation::Interpolation(Context &ctx,InterpolationConfig config) :
    Operator(ctx),
    config_(config)
{
    DLPRIM_CHECK((config_.scale_y <= 0) == (config_.scale_x <= 0));
    DLPRIM_CHECK((config_.out_h <= 0) == (config_.out_w <= 0));
    DLPRIM_CHECK((config_.scale_x <= 0) != (config_.out_w <= 0));
    DLPRIM_CHECK(config_.method == InterpolateType::bilinear || config_.align_corners == false);
}

Interpolation::~Interpolation()
{
}

int Interpolation::calc_size(int in,double scale)
{
    return int(in * scale);
}

Shape Interpolation::calc_size(Shape in_shape)
{
    Shape out_shape = in_shape;
    if(config_.out_h > 0) {
        out_shape[2] = config_.out_h;
        out_shape[3] = config_.out_w;
    }
    else {
        out_shape[2] = calc_size(in_shape[2],config_.scale_y);
        out_shape[3] = calc_size(in_shape[3],config_.scale_x);
    }
    return out_shape;
}

void Interpolation::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0].shape();
    DLPRIM_CHECK(in_shape.size() == 4);
    Shape out_shape=calc_size(in_shape);
    TensorSpecs outs(out_shape,in[0].dtype());
    out.assign({outs});
    p.clear();
    ws = 0;
}

void Interpolation::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    Shape in_shape = in[0];
    DLPRIM_CHECK(in_shape.size() == 4);
    Shape out_shape = calc_size(in_shape);
    out.assign({out_shape});
    ws = 0;
}

namespace {
    struct HandleFWD {
        static void handle(
                    float &y,
                    float &x00,float &x01,
                    float &x10,float &x11,
                    float w_r0, float w_r1,float w_c0,float w_c1)
        {
            y = w_r0 * (w_c0 * x00 + w_c1 *x01)
              + w_r1 * (w_c0 * x10 + w_c1 *x11);
        }
    };
    struct HandleBWD {
        static void handle(
                    float &y,
                    float &x00,float &x01,
                    float &x10,float &x11,
                    float w_r0, float w_r1,float w_c0,float w_c1)
        {
            float val = y;
            x00 += val * w_r0 * w_c0;
            x01 += val * w_r0 * w_c1;
            x10 += val * w_r1 * w_c0;
            x11 += val * w_r1 * w_c1;
        }
    };
}


float Interpolation::calc_bin_scale(float scale,int x_size,int y_size)
{
    if(config_.align_corners) {
        if(y_size <= 1)
            return 0;
        return double(x_size-1)/(y_size-1);
    }
    if(scale > 0)
        return 1.0/scale;
    return double(x_size)/y_size;
}

float Interpolation::calc_bin_src(int dst_index,float scale)
{
    if(config_.align_corners)
        return scale * dst_index;
    else
        return std::max(scale * (dst_index + 0.5f) - 0.5f,0.0f);
}

std::tuple<int,int,float,float> Interpolation::calc_bin_src_weight(int dst_intex,float scale,int size)
{
    float p0f = calc_bin_src(dst_intex,scale);
    int p0 = p0f;
    int dp = (p0 < size-1) ? 1 : 0;
    int p1 = p0 + dp;
    float w1 = p0f - p0;
    float w0 = 1 - w1;
    return std::make_tuple(p0,p1,w0,w1);
}

void Interpolation::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(input[0].shape().size()==4);
    DLPRIM_CHECK(output[0].shape() == calc_size(input[0].shape()));
    DLPRIM_CHECK(input[0].dtype() == output[0].dtype());
    {
        core::interpolate2d(input[0],output[0],config_.scale_y,config_.scale_x,config_.method,config_.align_corners,e);
    }
}

void Interpolation::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    if(!input[0].requires_gradient)
        return;
 
    DLPRIM_CHECK(input[0].diff.shape().size()==4);
    DLPRIM_CHECK(output[0].diff.shape() == calc_size(input[0].diff.shape()));
    DLPRIM_CHECK(input[0].diff.dtype() == output[0].diff.dtype());
    float accum = input[0].accumulate_gradient;

	{
        core::interpolate2d_backward(input[0].diff,output[0].diff,config_.scale_y,config_.scale_x,config_.method,config_.align_corners,accum,e);
    }
}

} // dlprim
