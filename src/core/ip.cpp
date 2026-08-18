///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>

namespace dlprim {
namespace core {
    class IPForwardImpl : public IPForward {
    public:
        IPForwardImpl(const tart::device_ptr& device,IPSettings const &cfg,bool bias,StandardActivations activation)
        {
            int batch = cfg.optimal_batch_size;
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                device, cfg.dtype, false, true,
                batch,cfg.outputs,cfg.inputs,
                (bias ? gpu::GEMM::bias_N : gpu::GEMM::no_bias),
                activation            
            ));
        }
        virtual void enqueue(Tensor &x,Tensor &w,Tensor *bias,Tensor &y)
        {
            int batch = x.shape()[0];
            int inps  = x.shape().size_no_batch();
            int outs  = y.shape()[1];
            int bias_offset = bias ? bias->device_offset() : 0;
			tart::buffer_ptr bias_buffer = bias ? bias->device_buffer() :  nullptr;
            gemm_->gemm(batch,outs,inps,
                    x.device_buffer(),x.device_offset(),inps,
                    w.device_buffer(),w.device_offset(),inps,
                    y.device_buffer(),y.device_offset(),outs,
                    bias_buffer,bias_offset,0.0f,
                    y.shape().total_size());
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };

    std::unique_ptr<IPForward> IPForward::create(const tart::device_ptr& device,IPSettings const &config,bool bias,StandardActivations activation)
    {
        std::unique_ptr<IPForward> r(new IPForwardImpl(device,config,bias,activation));
        return r;
    }




    class IPBackwardDataImpl : public IPBackwardData {
    public:
        IPBackwardDataImpl(const tart::device_ptr& device,IPSettings const &cfg)
        {
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                        device, cfg.dtype,false,false,
                        cfg.optimal_batch_size,cfg.inputs,cfg.outputs,
                        gpu::GEMM::no_bias,
                        StandardActivations::identity            
                        ));
        }
        virtual void enqueue(Tensor &dx,Tensor &M,Tensor &dy,float factor) 
        {
            int outputs = dy.shape()[1];
            int inputs  = dx.shape().size_no_batch();
            gemm_->gemm(dy.shape()[0],inputs,outputs,
                        dy.device_buffer(),
                        dy.device_offset(),
                        outputs,
                        M.device_buffer(),
                        M.device_offset(),
                        M.shape()[1],
                        dx.device_buffer(),
                        dx.device_offset(),
                        inputs,
                        nullptr,0,
                        factor,
                        dx.shape().total_size());
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };

    std::unique_ptr<IPBackwardData> IPBackwardData::create(const tart::device_ptr& device,IPSettings const &config)
    {
        std::unique_ptr<IPBackwardData> r(new IPBackwardDataImpl(device,config));
        return r;
    }

    class IPBackwardFilterImpl : public IPBackwardFilter {
    public:
        IPBackwardFilterImpl(const tart::device_ptr& device,IPSettings const &config)
        {
            gemm_ = std::move(gpu::GEMM::get_optimal_gemm(
                        device, config.dtype,true,false,
                        config.outputs,config.inputs,config.optimal_batch_size,
                        gpu::GEMM::no_bias,
                        StandardActivations::identity            
            ));
        }
        virtual void enqueue(Tensor &x,Tensor &dM,Tensor &dy,float factor)
        {
            int outputs = dy.shape()[1];
            int inputs = x.shape().size_no_batch();
            gemm_->gemm(outputs,inputs,dy.shape()[0],
                                dy.device_buffer(),
                                dy.device_offset(),
                                outputs,
                                x.device_buffer(),
                                x.device_offset(),
                                inputs,
                                dM.device_buffer(),
                                dM.device_offset(),
                                dM.shape()[1],
                                nullptr,0,
                                factor,
                                dM.shape().total_size());
        }
    private:
        std::unique_ptr<gpu::GEMM> gemm_;
    };
    
    std::unique_ptr<IPBackwardFilter> IPBackwardFilter::create(const tart::device_ptr& device,IPSettings const &config)
    {
        std::unique_ptr<IPBackwardFilter> r(new IPBackwardFilterImpl(device,config));
        return r;
    }
    
    void enqueueBackwardBiasFilter(Tensor& dy, Tensor& dw, float beta)
    {
		// Ok, this is gonna be dumb.
		tart::device_ptr device = tensorDevice(dy);
		const Shape& shape = dy.shape();
		DLPRIM_CHECK(dy.dtype() == dw.dtype());
		
		uint32_t features_(shape[1]);
		uint32_t rows_columns_(shape.size_no_batch() / shape[1]);
		
		// taken from constructor
		int total_size = shape[0] * rows_columns_;
		uint32_t wg_;
		uint32_t items_per_wi_;
		uint32_t wg2_;
		uint32_t items_per_wi2_;
		uint32_t size2_;
		
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().bwd_bias(device, dy.dtype());
		tart::kernel_ptr kernel2_ = nullptr;
		if(total_size > 256 * 16)
		{
			wg_ = 256;
			items_per_wi_ = 16;
			int reduce_1st = wg_ * items_per_wi_;
			size2_ = (total_size + reduce_1st - 1) / reduce_1st;
			if(size2_ >= 256)
				wg2_ = 256;
			else if(size2_ >= 128)
				wg2_ = 128;
			else
				wg2_ = 64;
			items_per_wi2_ = (size2_ + wg2_ - 1) / wg2_;
			kernel2_ = prog->getKernel("bwd_bias");
		}
		else {
			size2_ = 1;
			if(total_size <= 64)
				wg_ = 64;
			else if(total_size <= 128)
				wg_ = 128;
			else
				wg_ = 256;
			items_per_wi_ = (total_size + wg_ - 1) / wg_;
		}
		tart::kernel_ptr kernel_ = prog->getKernel("bwd_bias");
		
		DLPRIM_CHECK(features_ == int(dw.shape()[0]));
		
		std::vector<uint32_t> spec = {
			wg_, 1, 1, // local size
			items_per_wi_, // ITEMS_PER_WI
			rows_columns_ // SIZE_2D
		};
		
		if(kernel2_) // 2-stage
		{
			// Why require user to pass workspace buffer when we can just make one here? c:
			// Maybe in the future we bring it back, but for now I don't care.
			tart::buffer_ptr float_ws = device->allocateBuffer(features_ * size2_ * tart::dtypes::float32.size());
			std::vector<uint32_t> l({wg_, 1});
			std::vector<uint32_t> g = gpu::round_range(wg_ * size2_,features_,l);
			g[0] = g[0]/l[0];
			int p=0;
			kernel_->setArg(p++,features_);
			kernel_->setArg(p++,total_size);
			dy.set_arg(kernel_,p);
			kernel_->setArg(p++, float_ws);
			kernel_->setArg(p++, 0);
			kernel_->setArg(p++,size2_);
			kernel_->setArg(p++,0.0f);
			g.resize(3, 1);
			kernel_->enqueue(g, spec);
			p=0;
			kernel2_->setArg(p++,features_);
			kernel2_->setArg(p++,size2_);
			kernel2_->setArg(p++, float_ws);
			kernel2_->setArg(p++, 0);
			dw.set_arg(kernel2_, p); 
			kernel2_->setArg(p++, 1);
			kernel2_->setArg(p++, beta);
			
			std::vector<uint32_t> spec2 = {
				wg2_, 1, 1,
				items_per_wi2_,
				size2_
			};
			kernel2_->enqueue({1, features_}, {});
		}
		else {
			int norm_size = (total_size + items_per_wi_ - 1) / items_per_wi_;
			std::vector<uint32_t> l({wg_, 1});
			std::vector<uint32_t> g = gpu::round_range(norm_size,features_,l);
			g[0] = g[0]/l[0];
			g[1] = g[1]/l[1];

			int p=0;
			kernel_->setArg(p++,features_);
			kernel_->setArg(p++,total_size);
			dy.set_arg(kernel_,p);
			dw.set_arg(kernel_,p);
			kernel_->setArg(p++,1);
			kernel_->setArg(p++,beta);
			g.resize(3, 1);
			kernel_->enqueue(g, spec);
		}
	}

    void add_bias(Tensor &t,Tensor &bias)
    {
        DLPRIM_CHECK(t.shape().size() >= 2);
        DLPRIM_CHECK(t.shape()[1] == bias.shape().total_size());

		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().fwd_bias(tensorDevice(t), t.dtype());
        tart::kernel_ptr k = prog->getKernel("fwd_bias");

        Shape const &s = t.shape();
        int B = s[0];
        int F = s[1];
        int RC = 1;
        if(s.size() >= 3)
            RC *= s[2];
        if(s.size() >= 4)
            RC *= s[3];

        int p = 0;
        k->setArg(p++,B);
        k->setArg(p++,F);
        k->setArg(p++,RC);
        t.set_arg(k,p);
        bias.set_arg(k,p);
        k->enqueue({RC,F,B}, {1, 1, 1});
    }


} // core
} // dlprim

