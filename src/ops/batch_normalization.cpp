///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/batch_normalization.hpp>
#include <dlprim/json.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/ops/initialization.hpp>
#include <cmath>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
        BatchNorm::BatchNorm(Context &ctx,BatchNormConfig const &config,DataType dt) :
            Operator(ctx),
            config_(config),
            dtype_(dt)
        {
        }

        BatchNormConfig BatchNormConfig::from_json(json::value const &v) 
        {
            BatchNormConfig cfg;
            cfg.features = v.get<int>("features",cfg.features);
            cfg.eps = v.get<float>("eps",cfg.eps);
            cfg.momentum = v.get<float>("momentum",cfg.momentum);
            cfg.affine = v.get<bool>("affine",cfg.affine);
            cfg.use_global_stats = v.get<bool>("use_global_stats",cfg.use_global_stats);
            return cfg;
        }

        BatchNorm::~BatchNorm() {}
        void BatchNorm::setup(std::vector<TensorSpecs> const &in,
                                std::vector<TensorSpecs> &out,
                                std::vector<TensorSpecs> &parameters,
                                size_t &workspace)
        {
            DLPRIM_CHECK(in.size()==1);
            DLPRIM_CHECK(in[0].shape().size() >= 2);
            if(config_.features == -1) {
                config_.features = in[0].shape()[1];
            }
            else {
                DLPRIM_CHECK(config_.features ==  int(in[0].shape()[1]));
            }
            out = in;
            parameters.clear();
            Shape std_shape(config_.features);
            parameters.push_back(TensorSpecs(std_shape,dtype_,false)); // mean
            parameters.push_back(TensorSpecs(std_shape,dtype_,false)); // variance
            if(config_.affine) {
                parameters.push_back(TensorSpecs(std_shape,dtype_));
                parameters.push_back(TensorSpecs(std_shape,dtype_));
            }
            if(mode() != CalculationsMode::predict) {
                current_mean_ = Tensor(ctx_,std_shape,dtype_);
                current_var_ = Tensor(ctx_,std_shape,dtype_);
            }
            {
                bn_gpu_ = std::move(core::BatchNormFwdBwd::create(ctx_,in[0].shape(),dtype_));
                workspace = bn_gpu_->workspace();
            }
            setup_shape_ = in[0].shape();
        }
        void BatchNorm::mode(CalculationsMode m)
        {
            Operator::mode(m);
        }
        
        void BatchNorm::initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e)
        {
            set_to_zero(parameters.at(0),e);
            set_to_constant(parameters.at(1),1.0,e);
            if(config_.affine) {
                set_to_constant(parameters.at(2),1.0,e);
                set_to_zero(parameters.at(3),e);
            }
        }
        
        void BatchNorm::reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,size_t &ws)
        {
            out = in;
            if(in[0][0] > setup_shape_[0] || in[0].size_no_batch()!=setup_shape_.size_no_batch()) {
                setup_shape_ = in[0];
                {
                    bn_gpu_.reset();
                    bn_gpu_ = std::move(core::BatchNormFwdBwd::create(ctx_,in[0],dtype_));
                }
            }
            if(bn_gpu_) {
                ws = bn_gpu_->workspace();
            }
            else {
                ws = 2 * config_.features * sizeof(float);
            }
        }
		
        
        void BatchNorm::forward(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output,
                                  std::vector<Tensor> &parameters,
                                  Tensor &ws,
                                  ExecutionContext const &e)
        {
            {
                Tensor mean,var;
                ExecutionContext elast;
                if(mode() == CalculationsMode::train && !config_.use_global_stats) {
                    int M = input[0].shape().total_size() / config_.features;
                    bn_gpu_->enqueue_calculate_batch_stats(
                            input[0],
                            current_mean_,current_var_,
                            ws,e.generate_series_context(0,3));

                    bn_gpu_->enqueue_update_running_stats(
                            config_.momentum,(1.0f-config_.momentum),
                            current_mean_,parameters[0],
                            (config_.momentum * M) / (M-1),(1.0f-config_.momentum),
                            current_var_,parameters[1],
                            ws,e.generate_series_context(1,3));
                    mean = current_mean_;
                    var = current_var_;
                    elast = e.generate_series_context(2,3);
                }
                else  {
                    mean = parameters.at(0);
                    var  = parameters.at(1);
                    elast = e;
                }
                if(config_.affine) {
                    bn_gpu_->enqueue_forward_affine(
                            input[0],output[0],
                            parameters.at(2),parameters.at(3),
                            mean,var,
                            config_.eps,
                            ws,e.generate_series_context(2,3));
                }
                else {
                    bn_gpu_->enqueue_forward_direct(
                            input[0],output[0],
                            mean,var,
                            config_.eps,
                            ws,e.generate_series_context(2,3));
                }
            }
        }

        int BatchNorm::plane_size(Shape const &s)
        {
            size_t n=1;
            for(int i=2;i<s.size();i++)
                n*=s[i];
            return n;
        }

        void BatchNorm::compute_conv_parameters(Tensor &mean,Tensor &var,Tensor *at,Tensor *bt)
        {
            float *scale = combined_scale_.data<float>();
            float *bias  = combined_bias_.data<float>();
            float *m = mean.data<float>();
            float *v = var.data<float>();
            float *a = at ? at->data<float>() : nullptr;
            float *b = bt ? bt->data<float>() : nullptr;
            for(int i=0;i<config_.features;i++) {
                float alpha = 1.0f / std::sqrt(v[i] + config_.eps);
                float beta  = - m[i] * alpha;
                if(a && b) {
                    alpha *= a[i];
                    beta = a[i]*beta + b[i];
                }
                scale[i] = alpha;
                bias[i] = beta;
            }
        }

        void BatchNorm::update_sums(int M,Tensor &cm,Tensor &cv,Tensor &sm,Tensor &sv)
        {
            cblas_sscal(config_.features,(1.0f - config_.momentum),sm.data<float>(),1);
            cblas_saxpy(config_.features,config_.momentum,cm.data<float>(),1,sm.data<float>(),1);
            cblas_sscal(config_.features,(1.0f - config_.momentum),sv.data<float>(),1);
            float variance_factor = config_.momentum * M / (M - 1);
            cblas_saxpy(config_.features,variance_factor,cv.data<float>(),1,sv.data<float>(),1);
        }

        void BatchNorm::get_batch_stats(Tensor &x,Tensor &mean,Tensor &var)
        {
            float *m = mean.data<float>();
            float *v = var.data<float>();
            float *img = x.data<float>();
            size_t rc_size = plane_size(x.shape());
            int M = x.shape()[0]*rc_size;
            float factor = 1.0f / M;
            for(int f=0;f<config_.features;f++) {
                m[f] = 0;
                v[f] = 0;
                float s=0,s2=0;
                for(unsigned b=0;b<x.shape()[0];b++) {
                    float *ptr = img + (b*config_.features + f)*rc_size;
                    for(unsigned rc=0;rc < rc_size;rc++) {
                        float val = *ptr++;
                        s+= val;
                        s2 += val*val;
                    }
                }
                s*=factor;
                s2*=factor;
                m[f] = s;
                v[f] = s2 - s*s;
            }
        }

        void BatchNorm::backward( std::vector<TensorAndGradient> &input,
                                    std::vector<TensorAndGradient> &output,
                                    std::vector<TensorAndGradient> &parameters,
                                    Tensor &ws,
                                    ExecutionContext const &e)
        {
            {
                bool training = mode()==CalculationsMode::train;
                Tensor mean = training ? current_mean_ : parameters[0].data;
                Tensor var  = training ? current_var_  : parameters[1].data;
                if(config_.affine) {
                    bn_gpu_->enqueue_backward_affine(
                            training,
                            input[0].data,output[0].diff,
                            mean,var,
                            parameters.at(2).data,
                            (input[0].requires_gradient ? &input[0].diff : nullptr),
                            input[0].accumulate_gradient,
                            (parameters[2].requires_gradient ? &parameters[2].diff : nullptr),
                            parameters[2].accumulate_gradient,
                            (parameters[3].requires_gradient ? &parameters[3].diff : nullptr),
                            parameters[3].accumulate_gradient,
                            config_.eps,
                            ws,e);
                }
                else {
                    if(!input[0].requires_gradient)
                        return;
                    bn_gpu_->enqueue_backward_direct(
                            training,
                            input[0].data,output[0].diff,
                            mean,var,
                            input[0].diff,
                            input[0].accumulate_gradient,
                            config_.eps,
                            ws,e);
                }
            }
        }


} // namesapce
