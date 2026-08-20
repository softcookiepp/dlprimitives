///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/pool.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>
namespace dlprim {
namespace core {

	void pooling2dFwd(bool avg, std::array<uint32_t, 2> poolSize, std::array<uint32_t, 2> padSize, std::array<uint32_t, 2> strideSize, bool includePad, Tensor& in, Tensor& out)
	{
		tart::device_ptr device = tensorDevice(in);
		const tart::DType& dt = in.dtype();
		const uint32_t wg_size_ = 8;
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().pooling(device, dt);
		auto kernel_ = prog->getKernel("pooling");
		//auto bwd_kernel_ = prog->getKernel("pooling_bw");
		
		int bc = in.shape()[0]*in.shape()[1];

		int in_h = in.shape()[2];
		int in_w = in.shape()[3];

		int out_h = out.shape()[2];
		int out_w = out.shape()[3];

		int p=0;
		kernel_->setArg(p++,bc);
		kernel_->setArg(p++,in_h);
		kernel_->setArg(p++,in_w);
		kernel_->setArg(p++,out_h);
		kernel_->setArg(p++,out_w);
		in.set_arg(kernel_,p);
		out.set_arg(kernel_,p);

		std::vector<uint32_t> wg({wg_size_, wg_size_, 1});
		std::vector<uint32_t> gr = gpu::round_range(out_h,out_w,bc,wg);
		gr[0] = gr[0]/wg[0];
		gr[1] = gr[1]/wg[1];
		gr.resize(3, 1);
		kernel_->enqueue(gr, {
				wg_size_,
				poolSize[0],
				poolSize[1],
				strideSize[0],
				strideSize[1],
				padSize[0],
				padSize[1],
				static_cast<uint32_t>(avg),
				static_cast<uint32_t>(includePad)
			}
		);
	}
	
	void pooling2dBwd(bool avg, const std::array<uint32_t, 2>& poolSize, const std::array<uint32_t, 2>& padSize, const std::array<uint32_t, 2>& strideSize, bool includePad, Tensor* x, Tensor& dx, Tensor& dy, float factor)
	{
		tart::device_ptr device = tensorDevice(dx);
		const tart::DType& dt = dx.dtype();
		const uint32_t wg_size_ = 8;
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().pooling(device, dt);
		auto bwd_kernel_ = prog->getKernel("pooling_bw");
		
		int bc = dx.shape()[0]*dx.shape()[1];
		int in_h = dx.shape()[2];
		int in_w = dx.shape()[3];

		int out_h = dy.shape()[2];
		int out_w = dy.shape()[3];

		int p=0;

		scale_tensor(factor, dx);
		bwd_kernel_->setArg(p++,bc);
		bwd_kernel_->setArg(p++,in_h);
		bwd_kernel_->setArg(p++,in_w);
		bwd_kernel_->setArg(p++,out_h);
		bwd_kernel_->setArg(p++,out_w);
		if(x == nullptr)
		{
			// use placeholder
			dy.set_arg(bwd_kernel_,p);
		}
		else
		{
			x->set_arg(bwd_kernel_,p);
		}
		dy.set_arg(bwd_kernel_,p);
		// one for regular, one for atomic. The path taken will depend on spec constants
		dx.set_arg(bwd_kernel_,p);
		bwd_kernel_->setArg(p++, dx.device_buffer());

		std::vector<uint32_t> wg({wg_size_,wg_size_,1});
		std::vector<uint32_t> gr = gpu::round_range(out_h,out_w,bc,wg);
		for (size_t i = 0; i < wg.size(); i += 1)
			gr[i] = gr[i]/wg[i];
		gr.resize(3, 1);
		bwd_kernel_->enqueue(gr, {
				wg_size_,
				poolSize[0],
				poolSize[1],
				strideSize[0],
				strideSize[1],
				padSize[0],
				padSize[1],
				static_cast<uint32_t>(avg),
				static_cast<uint32_t>(includePad)
			}
		);
	}

	class Pooling2DFWBDImpl {
	public:
		size_t workspace() { return 0; }
		Pooling2DFWBDImpl(const tart::device_ptr& device,bool avg,int k[2],int p[2],int s[2],bool inc_pad, const tart::DType& dt) :
			mPoolSize({k[0], k[1]}),
			mStrideSize({s[0], s[1]}),
			mPadSize({p[0], p[1]}),
			mPoolMode(static_cast<uint32_t>(avg)),
			mIncludePad(inc_pad)
		{
			wg_size_ = 8;
			tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().pooling(device, dt);
			kernel_ = prog->getKernel("pooling");
			bwd_kernel_ = prog->getKernel("pooling_bw");
		}

		void forward(Tensor &in,Tensor &out)
		{
			pooling2dFwd(mPoolMode, {mPoolSize[0], mPoolSize[1]}, {mPadSize[0], mPadSize[1]}, {mStrideSize[0], mStrideSize[1]}, mIncludePad, in, out);
		}

		void backward(Tensor *x,Tensor &dx,Tensor &dy,float factor)
		{
			pooling2dBwd(mPoolMode, {mPoolSize[0], mPoolSize[1]}, {mPadSize[0], mPadSize[1]}, {mStrideSize[0], mStrideSize[1]}, mIncludePad, x, dx, dy, factor);
		}
	private:
		// spec constant parameters
		std::vector<uint32_t> mPoolSize;
		std::vector<uint32_t> mStrideSize;
		std::vector<uint32_t> mPadSize;
		uint32_t mPoolMode;
		uint32_t mIncludePad;
	
		int wg_size_;
		tart::kernel_ptr kernel_;
		tart::kernel_ptr bwd_kernel_;
	};

	class GlobalPoolingFWBWImpl  {
	public:
		GlobalPoolingFWBWImpl(const tart::device_ptr& device,bool avg,Shape const &sh, const tart::DType& dt = tart::dtypes::float32) 
		{
			avg_ = avg;
			DLPRIM_CHECK(sh.size() == 4);
			int sm_range = sh[2]*sh[3];
			if(sm_range <= 64)
				wg_size_ = 64;
			else if(sm_range <= 128)
				wg_size_ = 128;
			else 
				wg_size_ = 256;
			items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;
			tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().global_pooling(device, dt);
			kernel_ = prog->getKernel("global_pooling");
			kernel_bwd_ = prog->getKernel("global_pooling_bwd");
			sm_range_ = sm_range;
			int mpl = wg_size_ * items_per_wi_;
			nd_range_ = (sm_range_ + mpl - 1) / mpl * wg_size_;
		}

		void forward(Tensor &input,Tensor &output)
		{
			Shape in_shape = input.shape();
			int over = in_shape[2] * in_shape[3];
			DLPRIM_CHECK(over == sm_range_);
			int p=0;
			kernel_->setArg(p++,int(in_shape[0]*in_shape[1]));
			kernel_->setArg(p++,sm_range_);
			kernel_->setArg(p++,float(1.0f / (in_shape[2]*in_shape[3])));
			input.set_arg(kernel_,p);
			output.set_arg(kernel_,p);

			std::vector<uint32_t> gr({in_shape[0]*in_shape[1], nd_range_/wg_size_});
			//std::vector<uint32_t> wg({1, wg_size_});
			gr.resize(3, 1);
			kernel_->enqueue(gr, {wg_size_, items_per_wi_, static_cast<uint32_t>(avg_)});
		}
		void backward(Tensor *x,Tensor &dx,Tensor &dy, float factor)
		{
			Shape in_shape = dx.shape();
			int over = in_shape[2] * in_shape[3];
			DLPRIM_CHECK(over == sm_range_);
			int p=0;
			kernel_bwd_->setArg(p++,int(in_shape[0]*in_shape[1]));
			kernel_bwd_->setArg(p++,sm_range_);
			kernel_bwd_->setArg(p++,float(1.0f / (in_shape[2]*in_shape[3])));
			if(x != nullptr)
			{
				x->set_arg(kernel_bwd_,p);
			}
			else
			{
				// use dx as a placeholder hehe
				dx.set_arg(kernel_bwd_,p);
			}
			dx.set_arg(kernel_bwd_,p);
			dy.set_arg(kernel_bwd_,p);
			kernel_bwd_->setArg(p++,factor);

			//std::vector<uint32_t> wg(1, wg_size_);
			kernel_bwd_->enqueue({in_shape[0]*in_shape[1], nd_range_/wg_size_}, {wg_size_, items_per_wi_, static_cast<uint32_t>(avg_)});
		}
	private:
		tart::kernel_ptr kernel_;
		tart::kernel_ptr kernel_bwd_;
		int wg_size_;
		int items_per_wi_;
		int sm_range_;
		int nd_range_;
		bool avg_;
	};

	template<typename Impl>
	class ForwardImpl : public Pooling2DForward, public Impl {
	public:
		using Impl::Impl;
		size_t workspace() { return 0; }
		virtual void enqueue(Tensor &X,Tensor &Y)
		{
			this->forward(X,Y);
		}
	};

	template<typename Impl>
	class BackwardMax : public MaxPooling2DBackward, public Impl {
	public:
		using Impl::Impl;
		size_t workspace() { return 0; }
		virtual void enqueue(Tensor &X,Tensor &dX,Tensor &dY,float factor)
		{
			this->backward(&X,dX,dY,factor);
		}
	};

	template<typename Impl>
	class BackwardAvg : public AvgPooling2DBackward, public Impl {
	public:
		using Impl::Impl;
		size_t workspace() { return 0; }
		virtual void enqueue(Tensor &dX,Tensor &dY,float factor)
		{
			this->backward(nullptr,dX,dY,factor);
		}
	};

	std::unique_ptr<Pooling2DForward> Pooling2DForward::create_avg_pooling(const tart::device_ptr& device,int k[2],int p[2],int s[2],bool cip, const tart::DType& dt)
	{
		std::unique_ptr<Pooling2DForward> r(new ForwardImpl<Pooling2DFWBDImpl>(device,true,k,p,s,cip,dt));
		return r;
	}
   
	std::unique_ptr<Pooling2DForward> Pooling2DForward::create_global_max_pooling(const tart::device_ptr& device,Shape const &in_shape, const tart::DType& dt)
	{
		std::unique_ptr<Pooling2DForward> r(new ForwardImpl<GlobalPoolingFWBWImpl>(device,false,in_shape,dt));
		return r;
	}
	std::unique_ptr<Pooling2DForward> Pooling2DForward::create_global_avg_pooling(const tart::device_ptr& device, Shape const &in_shape, const tart::DType& dt)
	{
		std::unique_ptr<Pooling2DForward> r(new ForwardImpl<GlobalPoolingFWBWImpl>(device,true,in_shape,dt));
		return r;
	}

	std::unique_ptr<MaxPooling2DBackward>  MaxPooling2DBackward::create(const tart::device_ptr& device,int k[2],int p[2],int s[2], const tart::DType& dt)
	{
		std::unique_ptr<MaxPooling2DBackward> r(new BackwardMax<Pooling2DFWBDImpl>(device,false,k,p,s,false,dt));
		return r;
	}

	std::unique_ptr<AvgPooling2DBackward>  AvgPooling2DBackward::create(const tart::device_ptr& device,int k[2],int p[2],int s[2],bool cip, const tart::DType& dt)
	{
		std::unique_ptr<AvgPooling2DBackward> r(new BackwardAvg<Pooling2DFWBDImpl>(device,true,k,p,s,cip,dt));
		return r;
	}

	std::unique_ptr<MaxPooling2DBackward>  MaxPooling2DBackward::create_global(const tart::device_ptr& device,Shape const &s, const tart::DType& dt)
	{
		std::unique_ptr<MaxPooling2DBackward> r(new BackwardMax<GlobalPoolingFWBWImpl>(device,false,s,dt));
		return r;
	}

	std::unique_ptr<AvgPooling2DBackward>  AvgPooling2DBackward::create_global(const tart::device_ptr& device,Shape const &s, const tart::DType& dt)
	{
		std::unique_ptr<AvgPooling2DBackward> r(new BackwardAvg<GlobalPoolingFWBWImpl>(device,true,s,dt));
		return r;
	}


} // core
} // dlprim

