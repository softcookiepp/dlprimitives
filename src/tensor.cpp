///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/tensor.hpp>

#include <iostream>

namespace dlprim {
	struct Tensor::HostMem {
#if VULKAN_API
		std::vector<uint8_t> mHostMem;
#else
		void *p = nullptr;
#endif
		~HostMem()
		{
#if VULKAN_API
			// literally nothing...my frustration with legacy C++ practices grows indefinitely
#else
			free();
#endif
		}
		void free()
		{
#if VULKAN_API
			mHostMem.resize(0);
#else
			if(p) {
			#ifndef DLPRIM_WINDOWS
				::free(p); 
			#else
				_aligned_free(p);
			#endif
			}
			p = nullptr;
#endif
		}
		void alloc(
#if VULKAN_API
			size_t
#else
			cl_ulong
#endif
				size)
		{
#if VULKAN_API
			mHostMem.resize(size);
#else
			free();
            #ifndef DLPRIM_WINDOWS
            int r = posix_memalign(&p,128,size);
            if(r!=0) {
                free();
                throw std::bad_alloc();
            }
            #else
            p = _aligned_malloc(size,128);
            #endif
			if(!p)
				throw std::bad_alloc();
#endif
		}
	};
    Tensor::Tensor() :
        specs_(new TensorSpecs()),
        host_(new Tensor::HostMem()),
        cpu_tensor_(true),
        offset_(0),
        capacity_(0),full_capacity_(0)
    {
    }
    Tensor::Tensor(
#if VULKAN_API
			tart::buffer_ptr
#else
			cl::Buffer const &
#endif
			buffer,
			cl_ulong offset, Shape const &s, DataType d, bool is_train) :
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        cpu_tensor_(false),
        offset_(offset),
        capacity_(s.total_size()*size_of_data_type(d)),
        full_capacity_(capacity_ + offset * size_of_data_type(d))
    {
        buffer_ = buffer;
    }
    Tensor::Tensor(Context &ctx, Shape const &s,DataType d,bool is_train):
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        cpu_tensor_(ctx.is_cpu_context()),
        offset_(0),
        capacity_(s.total_size()*size_of_data_type(d)),
        full_capacity_(capacity_)
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
		if(cpu_tensor_)
			host_->alloc(size);
        if(!cpu_tensor_) {
#if VULKAN_API
			// gonna need to figure out where to put the device class
			throw std::runtime_error("buffer creation not implemented");
#else
            buffer_ = cl::Buffer(ctx.context(),CL_MEM_READ_WRITE,size);
#endif
        }
    }

    void Tensor::reshape(Shape const &new_shape)
    {
        if(new_shape.total_size() > capacity_)
            throw ValidationError("reshape: new size is larger than original");
        specs_->shape(new_shape);
    }

    void Tensor::to_device(ExecutionContext const &c,void *p,bool sync)
    {
        if(cpu_tensor_)
            memcpy(host_data(),p,memory_size());
        else
#if VULKAN_API
			buffer_->copyIn(p, memory_size(), offset_ * size_of_data_type(dtype()));
#else
            c.queue().enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype()), memory_size(), p,c.events(),c.event("write"));
#endif
    }

    void Tensor::to_device(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
#if VULKAN_API
		buffer_->copyIn(host_data(), memory_size(), offset_ * size_of_data_type(dtype()));
#else
        c.queue().enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype()), memory_size(), host_data(),c.events(),c.event("write"));
#endif
    }
    void Tensor::to_host(ExecutionContext const &c, void *p,bool sync)
    {
        if(cpu_tensor_) 
            memcpy(p,host_data(),memory_size());
        else
#if VULKAN_API
			buffer_->copyOut(p, memory_size(), offset_ * size_of_data_type(dtype()));
#else
            c.queue().enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype()), memory_size(), p,c.events(),c.event("read"));
#endif
    }
    void Tensor::to_host(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
#if VULKAN_API
		// all buffer copies in tart are sync, sorry :c
		buffer_->copyOut(host_data(), memory_size(), offset_ * size_of_data_type(dtype()));
#else
        c.queue().enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype()), memory_size(), host_data(),c.events(),c.event("read"));
#endif
    }

    Tensor Tensor::sub_tensor(size_t offset,Shape const &s,DataType d,bool trainable) const
    {
        size_t offset_bytes = offset * size_of_data_type(dtype());
        DLPRIM_CHECK(shape().total_size()*size_of_data_type(dtype()) >= s.total_size() * size_of_data_type(d));
        DLPRIM_CHECK((offset_ * size_of_data_type(dtype()) + offset_bytes) % size_of_data_type(d) == 0);
        Tensor r;
        r.specs_.reset(new TensorSpecs(s,d,trainable));
        r.cpu_tensor_ = cpu_tensor_;
        r.host_ = host_;
        r.buffer_ = buffer_;
        r.capacity_ = r.memory_size();
        r.full_capacity_ = full_capacity_;
        r.cpu_tensor_ = cpu_tensor_;
        r.offset_ = (offset_ * size_of_data_type(dtype())  + offset_bytes) / size_of_data_type(d);
        return r;
    }

    void *Tensor::host_data()
    {
#if VULKAN_API
		// for now, though I am not happy about this
		if(host_->mHostMem.size() == 0)
		{
			host_->alloc(full_capacity_);
		}
		return static_cast<char*>(host_->mHostMem.data()) + offset_ * size_of_data_type(dtype());
#else
		if(!host_->p) {
			host_->alloc(full_capacity_);
		}
        return static_cast<char*>(host_->p) + offset_ * size_of_data_type(dtype());
#endif
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
