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
    SliceCopy::SliceCopy(Context &ctx, const tart::DType& dtype) : mDtype(dtype)
    {
		mDevice = ctx.device();
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().copy(ctx.device(), mDtype);
        kernel_ = prog->getKernel("copy");
    }
    SliceCopy::~SliceCopy()
    {
    }
    void SliceCopy::tensor_slice_copy(int dim,size_t slice,
                           Tensor &target,size_t target_offset,
                           Tensor &source,size_t source_offset,
                           float scale)
    {
        Shape t = target.shape().split_and_merge_over_axis(dim);
        Shape s = source.shape().split_and_merge_over_axis(dim);
        DLPRIM_CHECK(target.tDtype() == mDtype);
        DLPRIM_CHECK(source.tDtype() == mDtype);
        DLPRIM_CHECK(s[0] == t[0]);
        DLPRIM_CHECK(source_offset + slice <= s[1]);
        DLPRIM_CHECK(target_offset + slice <= t[1]);
        DLPRIM_CHECK(s[2] == t[2]);
        int p = 0;
		kernel_->setArg(p++,uint32_t(slice));
        kernel_->setArg(p++,uint32_t(s[0]));
        kernel_->setArg(p++,uint32_t(t[1]));
        kernel_->setArg(p++,uint32_t(target_offset));
        kernel_->setArg(p++,uint32_t(s[1]));
        kernel_->setArg(p++,uint32_t(source_offset));
        kernel_->setArg(p++,uint32_t(s[2]));
        target.set_arg(kernel_,p);
        source.set_arg(kernel_,p);
        bind_as_dtype(kernel_,p,scale, mDtype);
        
        std::vector<uint32_t> totalInvocations({s[2], slice, s[0]});
        auto glPair = mDevice.lock()->chooseGlobalAndLocalSize(totalInvocations);
		kernel_->enqueue(glPair.first, glPair.second);
    }
}
}

