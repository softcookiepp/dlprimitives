///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/axpby.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>
#include <clblast_vk.h>

namespace dlprim {

AXPBY::AXPBY(Context &ctx,DataType dt) : ctx_(ctx)
{
    DLPRIM_CHECK(dt == float_data);
    tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().axpby(ctx.device());
    kernel_ = prog->getKernel("axpby");
}
AXPBY::~AXPBY()
{
}

void AXPBY::apply(float a,Tensor &x,float b,Tensor &y,Tensor &z,ExecutionContext const &e)
{
    DLPRIM_CHECK(x.shape().total_size() == y.shape().total_size());
    DLPRIM_CHECK(z.shape().total_size() == y.shape().total_size());
    size_t total = x.shape().total_size();
	//e.queue()->sync();
    {
        std::vector<uint32_t> l(1);
        if(total >= 256)
            l[0] = 256;
        else if(total >= 128)
            l[0] = 128;
        else
            l[0] = 64;

        std::vector<uint32_t> g = gpu::round_range(total,l);
        // because of course
        g[0] = g[0] / l[0];
        int p=0;
		kernel_->setArg(p++, uint64_t(total));
        kernel_->setArg(p++, a);
        x.set_arg(kernel_, p);
        kernel_->setArg(p++, b);
        y.set_arg(kernel_, p);
        z.set_arg(kernel_, p);
		kernel_->enqueue(g, l);
    }
}




} // dlprim
