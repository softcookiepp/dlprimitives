///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/concat.hpp>
#include <dlprim/core/common.hpp>
#include <my_cblas.hpp>
namespace dlprim {
    Concat::Concat(Context &ctx,ConcatConfig const &config) : Operator(ctx), cfg_(config)
    {
    }
    Concat::~Concat()
    {
    }
	void Concat::setup(std::vector<TensorSpecs> const &in,
                       std::vector<TensorSpecs> &out,
                       std::vector<TensorSpecs> &parameters,
                       size_t &workspace)
    {
        std::vector<Shape> ins,outs;
        dtype_ = in.at(0).dtype();
        DLPRIM_CHECK(dtype_==float_data);
        for(size_t i=0;i<in.size();i++) {
            ins.push_back(in[i].shape());
            DLPRIM_CHECK(in[0].dtype() == in[i].dtype());
        }
        reshape(ins,outs,workspace);
        out.clear();
        out.push_back(TensorSpecs(outs[0],in[0].dtype()));
        parameters.clear();
        {
            copy_.reset(new core::SliceCopy(ctx_,dtype_));
        }
    }
    void Concat::reshape(std::vector<Shape> const &in,
                         std::vector<Shape> &out,
                         size_t &ws)
    {
        size_t total_dim = 0;
        for(size_t i=0;i<in.size();i++) {
            DLPRIM_CHECK(cfg_.dim < in[i].size());
            total_dim += in[i][cfg_.dim];
            if(i == 0)
                continue;
            DLPRIM_CHECK(in[0].size() == in[i].size());
            for(int j=0;j<in[0].size();j++) {
                if(j==cfg_.dim)
                    continue;
                DLPRIM_CHECK(in[0][j] == in[i][j]);
            }
        }
        Shape r = in[0];
        r[cfg_.dim] = total_dim;
        out.assign({r});
        ws = 0;
    }
    
    struct AddBlock {
        void operator()(float *tgt,float *src,size_t len)
        {
            cblas_saxpy(len,1.0,src,1,tgt,1);
        }
    };
    struct AddOne {
        void operator()(float *tgt,float *src,size_t len)
        {
            *tgt += *src;
        }
    };
    struct FwdCopyBlock {
        void operator()(float *tgt,float *src,size_t len)
        {
            memcpy(tgt,src,sizeof(float)*len);
        }
    };

    struct FwdCopyOne {
        void operator()(float *tgt,float *src,size_t )
        {
            *tgt=*src;
        }
    };


    struct BwdCopyBlock {
        void operator()(float *tgt,float *src,size_t len)
        {
            memcpy(src,tgt,len*sizeof(float));
        }
    };

    struct BwdCopyBlockFactor {
        float f;
        BwdCopyBlockFactor(float v) : f(v) {}
        void operator()(float *tgt,float *src,size_t len)
        {
            cblas_sscal(len,f,src,1);
            cblas_saxpy(len,1.0f,tgt,1,src,1);
        }
    };
    struct BwdCopyOne {
        void operator()(float *tgt,float *src,size_t)
        {
            *src = *tgt;
        }
    };
    struct BwdCopyOneFactor {
        float f;
        BwdCopyOneFactor(float v) :f(v) {}
        void operator()(float *tgt,float *src,size_t)
        {
            *src = *src *f + *tgt;
        }
    };


	void Concat::forward(std::vector<Tensor> &input,
                         std::vector<Tensor> &output,
                         std::vector<Tensor> &,
                         Tensor &,
                         ExecutionContext const &q)
    {
        DLPRIM_CHECK(output.size()==1);
        bool copy_one = input.at(0).shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        size_t offset = 0;
        int dim = cfg_.dim;
        for(size_t i=0;i<input.size();i++) {
            {
                size_t slice = input[i].shape()[cfg_.dim];
                copy_->tensor_slice_copy(cfg_.dim,slice,
                                         output[0],offset,
                                         input[i],0,
                                         0.0f,q);
                offset += slice;
            }
        }
        DLPRIM_CHECK(offset == output[0].shape()[cfg_.dim]);
    }
    void Concat::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &q)
    {
        DLPRIM_CHECK(output.size()==1);
        bool copy_one = output.at(0).diff.shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        size_t offset = 0;
        int total = 0;
        int count = 0;
        for(auto const &inp : input)
            total += int(inp.requires_gradient);

        for(size_t i=0;i<input.size();i++)  {
            if(!input[i].requires_gradient) {
                offset += input[i].data.shape()[cfg_.dim];
                continue;
            }
            {
                size_t slice = input[i].diff.shape()[cfg_.dim];
                copy_->tensor_slice_copy(cfg_.dim,slice,
                                         input[i].diff,0,
                                         output[0].diff,offset,
                                         input[i].accumulate_gradient,
                                         q);
                offset += slice;
            }
        }
        DLPRIM_CHECK(offset == output[0].diff.shape()[cfg_.dim]);
    }

    Slice::Slice(Context &ctx,SliceConfig const &config) : Operator(ctx), cfg_(config)
    {
    }
    Slice::~Slice()
    {
    }
	void Slice::setup(std::vector<TensorSpecs> const &in,
                       std::vector<TensorSpecs> &out,
                       std::vector<TensorSpecs> &parameters,
                       size_t &workspace)
    {
        dtype_ = in.at(0).dtype();
        Shape in_shape = in[0].shape();
        DLPRIM_CHECK(dtype_==float_data);
        DLPRIM_CHECK(in_shape.size() > cfg_.dim);
        if(cfg_.end == -1)
            cfg_.end = in_shape[cfg_.dim];
        DLPRIM_CHECK(cfg_.begin >= 0);
        DLPRIM_CHECK(cfg_.begin < cfg_.end);
        DLPRIM_CHECK(cfg_.end <= int(in_shape[cfg_.dim]));

        Shape new_shape = in[0].shape();
        new_shape[cfg_.dim] = cfg_.end - cfg_.begin;
        out.assign({TensorSpecs(new_shape,dtype_)});

        parameters.clear();
        {
            copy_.reset(new core::SliceCopy(ctx_,dtype_));
            scale_.reset(new core::Scale(ctx_, data_type_to_tart_dtype(dtype_)));
        }
        workspace = 0;
    }
    void Slice::reshape(std::vector<Shape> const &in,
                         std::vector<Shape> &out,
                         size_t &ws)
    {
        DLPRIM_CHECK(in.size() == 1);
        DLPRIM_CHECK(in[0].size() > cfg_.dim);
        DLPRIM_CHECK(cfg_.end <= int(in[0][cfg_.dim]));
        out = in;
        out[0][cfg_.dim] = cfg_.end - cfg_.begin;
        ws = 0;
    }
	void Slice::forward(std::vector<Tensor> &input,
                         std::vector<Tensor> &output,
                         std::vector<Tensor> &,
                         Tensor &,
                         ExecutionContext const &q)
    {
        bool copy_one = input.at(0).shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        int dim = cfg_.dim;
        DLPRIM_CHECK(output.at(0).shape()[cfg_.dim] == unsigned(cfg_.end - cfg_.begin));
        {
            size_t slice = cfg_.end - cfg_.begin;
            copy_->tensor_slice_copy(cfg_.dim,slice,
                                     output[0],0,
                                     input[0],cfg_.begin,
                                     0.0f,q);
        }
    }
    void Slice::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &q)
    {
        DLPRIM_CHECK(output.size()==1);
        bool copy_one = output.at(0).diff.shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        if(!input[0].requires_gradient)
            return;

        {
            int total = (input[0].accumulate_gradient != 1.0) + 1;
            int index = 0;
            if(input[0].accumulate_gradient != 1.0)
                scale_->enqueue(input[0].accumulate_gradient,input[0].diff,q);

            size_t slice = cfg_.end - cfg_.begin;
            copy_->tensor_slice_copy(cfg_.dim,slice,
                                     input[0].diff,cfg_.begin,
                                     output[0].diff,0,
                                     1.0f,
                                     q);
        }
    }


    
} // dlprim
