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
	
	void globalPoolingFwd(bool avg, Tensor& input, Tensor& output)
	{
		tart::device_ptr device = tensorDevice(input);
		const tart::DType& dt = input.dtype();
		Shape in_shape = input.shape();
		DLPRIM_CHECK(in_shape.size() == 4);
		uint32_t sm_range = input.shape()[2]*input.shape()[3];
		uint32_t wg_size_;
		if(sm_range <= 64)
			wg_size_ = 64;
		else if(sm_range <= 128)
			wg_size_ = 128;
		else 
			wg_size_ = 256;
		uint32_t items_per_wi_ = (sm_range + wg_size_ - 1) / wg_size_;
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().global_pooling(device, dt);
		auto kernel_ = prog->getKernel("global_pooling");
		//auto kernel_bwd_ = prog->getKernel("global_pooling_bwd");

		uint32_t mpl = wg_size_ * items_per_wi_;
		uint32_t nd_range_ = (sm_range + mpl - 1) / mpl * wg_size_;
		
		int p=0;
		kernel_->setArg(p++, int(in_shape[0]*in_shape[1]));
		kernel_->setArg(p++, sm_range);
		kernel_->setArg(p++, float(1.0f / (in_shape[2]*in_shape[3])));
		input.set_arg(kernel_, p);
		output.set_arg(kernel_, p);

		std::vector<uint32_t> gr({in_shape[0]*in_shape[1], nd_range_/wg_size_});
		//std::vector<uint32_t> wg({1, wg_size_});
		gr.resize(3, 1);
		kernel_->enqueue(gr, {wg_size_, items_per_wi_, static_cast<uint32_t>(avg)});
	}
} // core
} // dlprim

