///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <clblast_vk.h>
#include <iostream>

enum class RandomType
{
	eUniform = 1,
	eBernoulli = 2,
	eNormal = 3
};

namespace dlprim {
namespace core {
    Scale::Scale(Context &ctx,DataType dt)
    {
        DLPRIM_CHECK(dt==float_data);
		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx,"scal");
		k_ = prog->getKernel("sscal");
    }
    void Scale::enqueue(float s,Tensor &t,ExecutionContext const &ec)
    {
		Context ctx(ec);
		#if 1
			clblast::Scal<float>(t.shape().total_size(), s, t.device_buffer(), t.device_offset(), 1, ctx.device());
		#else
			int p = 0;
			size_t size = t.shape().total_size();
			int wg;
			if(size >= 1024)
				wg = 256;
			else
				wg = 64;
			k_->setArg(p++, uint32_t(size));
			k_->setArg(p++, s);
			t.set_arg(k_,p);
			std::vector<uint32_t> l({wg, 1, 1});
			std::vector<uint32_t> g = gpu::round_range(size,l);
			g[0] = g[0]/l[0];
			g.resize(3, 1);
			k_->enqueue(g, l);
        #endif
    }
    
    void scale_tensor(float s,Tensor &t,ExecutionContext const &ec)
    {
        Context ctx(ec);
		Scale sc(ctx,t.dtype());
		sc.enqueue(s,t,ec);
    }

    ///
    /// Set to zero tensor - OpenCL only
    ///
    void fill_tensor(Tensor &t,double value,ExecutionContext const &e)
    {
        Context ctx(e);
        pointwise_operation({},{t},{value},"y0=dtype(w0);",e);
    }

    void fill_random(Tensor &t, uint64_t philox_seed, uint64_t philox_seq,RandomDistribution dist,float p1,float p2,ExecutionContext const &e)
    {
        Context ctx(e);
        DLPRIM_CHECK(t.dtype() == float_data);
		tart::program_ptr prog = gpu::Cache::instance().get_program(
			ctx, "random", "dtype", data_type_to_opencl_type(t.dtype()));
		tart::kernel_ptr k = prog->getKernel("fill");
		uint64_t total = t.shape().total_size();
        int p=0;
		k->setArg(p++,total);
        t.set_arg(k,p);
        k->setArg(p++,philox_seed);
        k->setArg(p++,philox_seq);
        k->setArg(p++,p1);
        k->setArg(p++,p2);
        
        const size_t globalSizeX = (total+3)/4;
        auto metadata = ctx.device()->getMetadata();
        const size_t maxWgX = metadata.physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
        size_t targetLocalSizeX = maxWgX;
        while (targetLocalSizeX > 1)
        {
			if (globalSizeX % targetLocalSizeX == 0) break;
			targetLocalSizeX -= 1;
		}
        const uint32_t adjustedGlobalSizeX = globalSizeX / targetLocalSizeX;
        k->enqueue({adjustedGlobalSizeX, 1, 1}, {targetLocalSizeX, 1, 1, (uint32_t)dist});
    }

} // core
} // dlprim
