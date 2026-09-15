///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <dlprim/core/util.hpp>
#include <dlprim/core/loss.hpp>
#include <iostream>

namespace dlprim {
namespace core {

    void softmax_backward(Tensor &dx,Tensor &y,Tensor &dy,bool log_softmax,float factor)
    {
#if 1
		throw std::runtime_error("Don't use this function, it is very broken at the moment");
#else
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

		tart::program_ptr prog = gpu::Cache::instance().get_program(dx.device_buffer()->getDevice(), "softmax",
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
#endif
    }

    ///
    /// Compute forward Negative log likelehood loss x should be log of prob
    ///
    void nll_loss_forward(Tensor &x,Tensor &lbl,Tensor &y,bool reduce,float scale)
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
        
		tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().nll_loss_fwd(tensorDevice(x), x.dtype(), lbl.dtype());
		
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
    void nll_loss_backward(Tensor &dx,Tensor &lbl,Tensor &dy,bool reduce,float scale,float factor)
    {
        DLPRIM_CHECK(dx.shape().size() == 2);
        DLPRIM_CHECK(dy.shape()==(reduce ? Shape(1) : Shape(dx.shape()[0])));
        DLPRIM_CHECK(dy.dtype() == dx.dtype());

        tart::program_ptr prog = gpu::PerDeviceProgramCache::instance().nll_loss_bwd(tensorDevice(dx), dx.dtype(), lbl.dtype());
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
	
	void softmaxAttempt2(Tensor x, Tensor y, std::vector<int> dims, bool useLogSoftmax)
	{
		tart::device_ptr device = tensorDevice(x);
		
		// broadcast tensors
		std::vector<Tensor> broadcasted({x, y});
		x = broadcasted[0];
		y = broadcasted[1];
		Shape xShape = x.shape();
		dims = getReduceDims(xShape, dims);
		
		// ensure dimensions aren't out of bounds
		for (size_t i = 0; i < dims.size(); i += 1)
			DLPRIM_CHECK(dims[i] < x.shape().size());
		
		// convert it to shape so that it can be bound
		Shape reduceDimShape = Shape::from_range(dims.begin(), dims.end());
		uint32_t numReduceElems = 1;
		// The way the kernel is set up, it needs the reduction shape to be provided as if a reduction operation is being performed,
		// even though no output is reduced.
		Shape dummyReduceShape = xShape;
		for (size_t i = 0; i < reduceDimShape.size(); i += 1)
		{
			numReduceElems *= xShape[reduceDimShape[i]];
			dummyReduceShape[reduceDimShape[i]] = 1;
		}
		
		tart::kernel_ptr k = gpu::PerDeviceProgramCache::instance().softmax(device, x.dtype(), y.dtype())->getKernel("main");;
		if (!k) throw std::runtime_error("suitable kernel not found");
		
		int p = 0;
		k->setArg(p++, x.device_buffer());
		k->setArg(p++, x.device_offset());
		bind_shape(k, p, x.stride());

		k->setArg(p++, y.device_buffer());
		k->setArg(p++, y.device_offset());
		bind_shape(k, p, y.stride());
		
		bind_shape(k, p, xShape);
		bind_shape(k, p, reduceDimShape);
		
		// calculate local size and work per thread, based on the max amount of local invocations along the X axis for this device
		uint32_t wpt = 1;
		uint32_t wgxSize = numReduceElems;
		uint32_t maxWgxSize = device->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
		while (wgxSize > maxWgxSize)
		{
			wpt += 1;
			wgxSize = numReduceElems / wpt;
			if (wgxSize == 0 || numReduceElems % wpt > 0) wgxSize += 1;
		}
		// need to ensure this is invoked at all
		if (wgxSize == 0 || numReduceElems % wpt > 0) wgxSize += 1;
		
		uint32_t localMemSize = wgxSize;
		#if 0
			// this is supposed to reduce the amount of local memory required, but for some reason its is causes the kernel to compute nan.
			// Still need to figure out why.
			if (device->getMetadata().subgroupAdd)
			{
				// Less local memory is required if subgroup arithmetic reduction is used
				uint32_t subgroupSize = device->getMetadata().maxSubgroupSize;
				localMemSize = localMemSize / subgroupSize;
				if (localMemSize == 0 || localMemSize % subgroupSize > 0) localMemSize += 1;
			}
		#endif
		
		std::vector<uint32_t> global = calcStridedTensorRange(device, dummyReduceShape);
		auto glPair = calcStridedTensorInvocations(device, dummyReduceShape);
		std::vector<uint32_t> spec = {
			wgxSize,
			static_cast<uint32_t>(xShape.size()),
			static_cast<uint32_t>(reduceDimShape.size()),
			numReduceElems,
			wpt,
			localMemSize,
			useLogSoftmax ? 1 : 0
		};
		k->enqueue(global, spec);
	}
    
} // core
} // dlprim

