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
	struct Tensor::HostMem
	{
		// TODO: maybe take into account mappable buffers somehow.
		// This would save us the trouble of allocating host memory in some circumstances
		std::vector<uint8_t> mHostMem;
		~HostMem()
		{
			// literally nothing...my frustration with legacy C++ practices grows indefinitely
		}
		void free()
		{
			mHostMem.resize(0);
		}
		void alloc(size_t size)
		{
			mHostMem.resize(size);
		}
	};
    Tensor::Tensor() :
        specs_(new TensorSpecs()),
        host_(new Tensor::HostMem()),
        offset_(0),
        capacity_(0),full_capacity_(0)
    {
    }
    Tensor::Tensor(
			tart::buffer_ptr
			buffer,
			uint64_t offset, 
			Shape const &s, DataType d, bool is_train) :
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        offset_(offset),
        capacity_(s.total_size()*size_of_data_type(d)),
        full_capacity_(capacity_ + offset * size_of_data_type(d))
    {
        buffer_ = buffer;
    }
    Tensor::Tensor(Context &ctx, Shape const &s,DataType d,bool is_train):
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        offset_(0),
        capacity_(s.total_size()*size_of_data_type(d)),
        full_capacity_(capacity_)
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
		buffer_ = ctx.device()->allocateBuffer(size);
		dev_ = ctx.device();
		own_buffer_ = true;
    }

	// tart makes you free buffers automatically. this behavior might change in the future.
	Tensor::~Tensor()
	{
		//dev_.lock()->deallocateBuffer(buffer_);
	}

    void Tensor::reshape(Shape const &new_shape)
    {
        if(new_shape.total_size() > capacity_)
            throw ValidationError("reshape: new size is larger than original");
        specs_->shape(new_shape);
    }

    void Tensor::to_device(ExecutionContext const &c,void *p,bool sync)
    {
		buffer_->copyIn(p, memory_size(), offset_ * size_of_data_type(dtype()));
    }

    void Tensor::to_device(ExecutionContext const &c,bool sync)
    {
		buffer_->copyIn(host_data(), memory_size(), offset_ * size_of_data_type(dtype()));
    }
    void Tensor::to_host(ExecutionContext const &c, void *p,bool sync)
    {
        {
			buffer_->copyOut(p, memory_size(), offset_ * size_of_data_type(dtype()));
		}
    }
    void Tensor::to_host(ExecutionContext const &c,bool sync)
    {
		// all buffer copies in tart are sync, sorry :c
		buffer_->copyOut(host_data(), memory_size(), offset_ * size_of_data_type(dtype()));
    }

    Tensor Tensor::sub_tensor(size_t offset,Shape const &s,DataType d,bool trainable) const
    {
        size_t offset_bytes = offset * size_of_data_type(dtype());
        DLPRIM_CHECK(shape().total_size()*size_of_data_type(dtype()) >= s.total_size() * size_of_data_type(d));
        DLPRIM_CHECK((offset_ * size_of_data_type(dtype()) + offset_bytes) % size_of_data_type(d) == 0);
        Tensor r;
        r.specs_.reset(new TensorSpecs(s,d,trainable));
        r.host_ = host_;
        r.buffer_ = buffer_;
        r.capacity_ = r.memory_size();
        r.full_capacity_ = full_capacity_;
        r.offset_ = (offset_ * size_of_data_type(dtype())  + offset_bytes) / size_of_data_type(d);
        return r;
    }

    void *Tensor::host_data()
    {
		// for now, though I am not happy about this
		if(host_->mHostMem.size() == 0)
		{
			host_->alloc(full_capacity_);
		}
		return (char*)(host_->mHostMem.data()) + offset_ * size_of_data_type(dtype());
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
