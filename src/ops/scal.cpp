///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/scal.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <dlprim/tensor.hpp>
#include <my_cblas.hpp>
namespace dlprim {
    Scal::Scal(Context &ctx, const tart::DType& dt) : ctx_(ctx)
    {
        tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().scal(ctx.device(), dt);
        k_ = prog->getKernel("sscal");
    }
    Scal::~Scal(){}
    
    void Scal::scale(float s,Tensor &t,ExecutionContext const &ec)
    {
        {
            int p = 0;
            size_t size = t.shape().total_size();
            int wg;
            if(size >= 1024)
                wg = 256;
            else
                wg = 64;
			k_->setArg(p++, uint32_t(size));
            k_->setArg(p++,s);
            t.set_arg(k_,p);
            std::vector<uint32_t> l({wg, 1, 1});
            std::vector<uint32_t> g = gpu::round_range(size,l);
            g[0] = g[0] / l[0];
            k_->enqueue(g, l);
        }
    }

}
