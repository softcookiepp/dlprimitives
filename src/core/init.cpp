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
#include <dlprim/gpu/tiered_cache.hpp>
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
    
    void scale_tensor(float s,Tensor &t)
    {
		clblast::Scal<float>(t.shape().total_size(), s, t.device_buffer(), t.device_offset(), 1, tensorDevice(t));
    }

    ///
    /// Set to zero tensor - OpenCL only
    ///
    void fill_tensor(Tensor &t,double value)
    {
        pointwise_operation({}, {t}, {value}, "y0 = dtype(w0);");
    }

    void fill_random(Tensor &t, uint64_t philox_seed, uint64_t philox_seq,RandomDistribution dist,float p1,float p2)
    {
        tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().random(tensorDevice(t), t.dtype());
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
        auto metadata = tensorDevice(t)->getMetadata();
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
