///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    void softmax_forward(Tensor &x,Tensor &y,bool log_softmax,ExecutionContext const &e)
    {
        DLPRIM_CHECK(x.shape().size() == 2 || x.shape().size() == 3);
        DLPRIM_CHECK(x.dtype() == tart::dtypes::float32);
        DLPRIM_CHECK(y.shape()==x.shape());
        DLPRIM_CHECK(y.dtype() == x.dtype());
        uint32_t sm_range=x.shape()[1];
#if 1
		// band-aid solution over a bigger problem.
		// need to find out why the kernel isn't working.
		uint32_t wg_size = 1;
#else
        uint32_t wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
#endif
        
        uint32_t items_per_wi = (sm_range + wg_size - 1) / wg_size;
        if (items_per_wi == 0) items_per_wi += 1;
		std::cout << "items per wi: " << items_per_wi << std::endl;
        uint32_t mpl = wg_size * items_per_wi;
        if (mpl == 0) mpl += 1;
        uint32_t nd_range = (sm_range + mpl - 1) / mpl * wg_size;
        Context ctx(e);
#if VULKAN_API
		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx.device(), "softmax",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "LOG_SM",int(log_softmax));
        tart::kernel_ptr kernel = prog->getKernel("softmax");
        Shape in_shape = x.shape();
        uint32_t b0 = in_shape[0];
        uint32_t b2 = in_shape.size() == 3 ? in_shape[2] : 1;
        int p = 0;
        kernel->setArg(p++, b0);
        kernel->setArg(p++, sm_range);
        kernel->setArg(p++, b2);
        x.set_arg(kernel, p);
        y.set_arg(kernel, p);

        //std::vector<uint32_t> wg({1,wg_size,1});
        //kernel->enqueue(gr, wg);
        kernel->enqueue({b0, nd_range/wg_size, b2}, {});
#else
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx.device(), "softmax",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "LOG_SM",int(log_softmax));
        cl::Kernel kernel(prog,"softmax");
        Shape in_shape = x.shape();
        int b0 = in_shape[0];
        int b2 = in_shape.size() == 3 ? in_shape[2] : 1;
        int p = 0;
        kernel.setArg(p++,b0);
        kernel.setArg(p++,sm_range);
        kernel.setArg(p++,b2);
        x.set_arg(kernel,p);
        y.set_arg(kernel,p);

        cl::NDRange gr(b0,nd_range,b2);
        cl::NDRange wg(1,wg_size,1);
        e.queue().enqueueNDRangeKernel(kernel,cl::NullRange,gr,wg,e.events(),e.event("softmax"));
#endif
    }

    void softmax_backward(Tensor &dx,Tensor &y,Tensor &dy,bool log_softmax,float factor,ExecutionContext const &e)
    {
        DLPRIM_CHECK(dx.shape().size() == 2 || dx.shape().size() == 3);
        DLPRIM_CHECK(dx.dtype() == tart::dtypes::float32);
        DLPRIM_CHECK(dy.shape() == dx.shape());
        DLPRIM_CHECK(dy.dtype() == dx.dtype());
        DLPRIM_CHECK(y.shape() == dx.shape());
        DLPRIM_CHECK(y.dtype() == dx.dtype());

        int sm_range=dx.shape()[1];

        int wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
        
        int items_per_wi = (sm_range + wg_size - 1) / wg_size;
		if (items_per_wi == 0) items_per_wi += 1;
        int mpl = wg_size * items_per_wi;
        int nd_range = (sm_range + mpl - 1) / mpl * wg_size;
        Context ctx(e);

		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx.device(), "softmax",
                            "WG_SIZE",wg_size,
                            "ITEMS_PER_WI",items_per_wi,
                            "LOG_SM",int(log_softmax));
        tart::kernel_ptr kernel = prog->getKernel("softmax_backward");
        Shape in_shape = dx.shape();
        int b0 = in_shape[0];
        int b2 = in_shape.size() == 3 ? in_shape[2] : 1;
        int p = 0;
        kernel->setArg(p++, (uint32_t)b0);
        kernel->setArg(p++, (uint32_t)sm_range);
        kernel->setArg(p++, (uint32_t)b2);
        dx.set_arg(kernel, p);
        y.set_arg(kernel, p);
        dy.set_arg(kernel, p);
        kernel->setArg(p++, factor);

        //std::vector<uint32_t> wg({1,wg_size,1});
        //kernel->enqueue(gr, wg);
        kernel->enqueue({b0,nd_range/wg_size,b2}, {});
    }

    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_forward(Tensor &x,Tensor &lbl,Tensor &y,bool reduce,float scale,ExecutionContext const &e)
    {
        DLPRIM_CHECK(x.shape().size() == 2);
        DLPRIM_CHECK(y.shape()==(reduce ? Shape(1) : Shape(x.shape()[0])));
        DLPRIM_CHECK(y.dtype() == x.dtype());
        int sm_range=x.shape()[0];

        int wg_size;
        if(sm_range <= 64)
            wg_size = 64;
        else if(sm_range <= 128)
            wg_size = 128;
        else 
            wg_size = 256;
        
        int items_per_wi = (sm_range + wg_size - 1) / wg_size;
        
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().nll_loss_fwd(tensorDevice(x), x.tDtype(), lbl.tDtype());
		
        tart::kernel_ptr kernel = prog->getKernel("nll_loss_forward");
        Shape in_shape = x.shape();
        int p = 0;
        kernel->setArg(p++,int(in_shape[0]));
        kernel->setArg(p++,int(in_shape[1]));
        x.set_arg(kernel,p);
        lbl.set_arg(kernel,p);
        y.set_arg(kernel,p);
        kernel->setArg(p++,scale);
        kernel->enqueue({1, 1, 1}, {wg_size, items_per_wi, static_cast<uint32_t>(reduce)});
    }
    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_backward(Tensor &dx,Tensor &lbl,Tensor &dy,bool reduce,float scale,float factor,ExecutionContext const &e)
    {
        DLPRIM_CHECK(dx.shape().size() == 2);
        DLPRIM_CHECK(dy.shape()==(reduce ? Shape(1) : Shape(dx.shape()[0])));
        DLPRIM_CHECK(dy.dtype() == dx.dtype());

        tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().nll_loss_bwd(tensorDevice(dx), dx.tDtype(), lbl.tDtype());
        tart::kernel_ptr kernel = prog->getKernel("nll_loss_backward");
        Shape in_shape = dx.shape();
        int p = 0;
        kernel->setArg(p++,int(in_shape[0]));
        kernel->setArg(p++,int(in_shape[1]));
        dx.set_arg(kernel,p);
        lbl.set_arg(kernel,p);
        dy.set_arg(kernel,p);
        kernel->setArg(p++,scale);
        kernel->setArg(p++,factor);
        kernel->enqueue({in_shape[1],in_shape[0], 1}, {1, 1, 1, static_cast<uint32_t>(reduce)} );
    }

    
} // core
} // dlprim

