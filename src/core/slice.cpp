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
namespace dlprim {
namespace core {
    
    void tensorSliceCopy(int dim,size_t slice,
		Tensor &target,size_t target_offset,
		Tensor &source,size_t source_offset,
		float scale)
    {
		Shape t = target.shape().split_and_merge_over_axis(dim);
        Shape s = source.shape().split_and_merge_over_axis(dim);
		DLPRIM_CHECK(target.dtype() == source.dtype());
        DLPRIM_CHECK(s[0] == t[0]);
        DLPRIM_CHECK(source_offset + slice <= s[1]);
        DLPRIM_CHECK(target_offset + slice <= t[1]);
        DLPRIM_CHECK(s[2] == t[2]);
		
		tart::device_ptr device = tensorDevice(target);
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().copy(device, target.dtype());
		tart::kernel_ptr k = prog->getKernel("copy");
		
		int p = 0;
		k->setArg(p++,uint32_t(slice));
        k->setArg(p++,uint32_t(s[0]));
        k->setArg(p++,uint32_t(t[1]));
        k->setArg(p++,uint32_t(target_offset));
        k->setArg(p++,uint32_t(s[1]));
        k->setArg(p++,uint32_t(source_offset));
        k->setArg(p++,uint32_t(s[2]));
        target.set_arg(k,p);
        source.set_arg(k,p);
        bind_as_dtype(k,p,scale, target.dtype());
        
        std::vector<uint32_t> totalInvocations({s[2], slice, s[0]});
        auto glPair = device->chooseGlobalAndLocalSize(totalInvocations);
		k->enqueue(glPair.first, glPair.second);
	}
}
}

