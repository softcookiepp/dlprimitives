///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    void activation_forward(Tensor &x,Tensor &y,StandardActivations activation, ExecutionContext const &ec)
    {
        Context ctx(ec);
#if VULKAN_API
		tart::program_ptr
#else
        cl::Program const &
#endif
			prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
#if VULKAN_API
        tart::kernel_ptr k = prog->getKernel("activation");
#else
        cl::Kernel k(prog,"activation");
#endif
        int p=0;
#if VULKAN_API
		uint32_t size = x.shape().total_size();
        k->setArg(p++,size);
        x.set_arg(k,p);
        y.set_arg(k,p);
#else
        cl_ulong size = x.shape().total_size();
        k.setArg(p++,size);
        x.set_arg(k,p);
        y.set_arg(k,p);
#endif
#if VULKAN_API
		std::vector<uint32_t> wg({256, 1, 1});
		std::vector<uint32_t> gr = gpu::round_range(size, wg);
		for (size_t i = 0; i < gr.size(); i += 1)
			gr[i] = gr[i] / wg[i];
		k->run(gr, wg);
#else
        cl::NDRange wg(256);
        cl::NDRange gr=gpu::round_range(size,wg);
        ec.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,wg,ec.events(),ec.event("activation"));
#endif
    }
    void activation_backward(Tensor &dx,Tensor &dy,Tensor &y,StandardActivations activation,float beta,ExecutionContext const &ec)
    {
        Context ctx(ec);
#if VULKAN_API
		tart::program_ptr const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        tart::kernel_ptr k = prog->getKernel("activation_diff");
        
        int p=0;
        uint64_t size = y.shape().total_size();
        k->setArg(p++,size);
        y.set_arg(k,p);
        dy.set_arg(k,p);
        dx.set_arg(k,p);
        k->setArg(p++,beta);

        std::vector<uint32_t> wg({256, 1, 1});
        std::vector<uint32_t> gr=gpu::round_range(size,wg);
        gr[0] = gr[0]/wg[0];
        k->run(gr, wg);
#else
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        cl::Kernel k(prog,"activation_diff");
        
        
        int p=0;
        cl_ulong size = y.shape().total_size();
        k.setArg(p++,size);
        y.set_arg(k,p);
        dy.set_arg(k,p);
        dx.set_arg(k,p);
        k.setArg(p++,beta);

        cl::NDRange wg(256);
        cl::NDRange gr=gpu::round_range(size,wg);
        ec.queue().enqueueNDRangeKernel(k,cl::NullRange,gr,wg,ec.events(),ec.event("activation_diff"));
#endif
    }

} // core
} // dlprim
