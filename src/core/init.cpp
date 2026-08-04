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
    Scale::Scale(Context &ctx, DataType dt)
    {
		// TODO: make this not just float
        DLPRIM_CHECK(dt==float_data);
    }
    void Scale::enqueue(float s,Tensor &t,ExecutionContext const &ec)
    {
		Context ctx(ec);
		clblast::Scal<float>(t.shape().total_size(), s, t.device_buffer(), t.device_offset(), 1, ctx.device());
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
        pointwise_operation({},{t},{value},"y0 = dtype(w0);",e);
    }

    void fill_random(Tensor &t, uint64_t philox_seed, uint64_t philox_seq,RandomDistribution dist,float p1,float p2,ExecutionContext const &e)
    {
        Context ctx(e);
        //DLPRIM_CHECK(t.dtype() == float_data);
		tart::program_ptr prog = gpu::Cache::instance().get_program(
			ctx, "random", "dtype", data_type_to_tart_dtype(t.dtype()).glsl());
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
