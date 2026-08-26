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
	
	bool geometryIsContiguous(const Shape& sizes, const Shape& strides)
	{
		// check for overflow, maaaybe later
		//assert(!overflows<std::int64_t>(sizes.size()));
		size_t dim = sizes.size();
		DLPRIM_CHECK(dim <= sizes.getRawShape().size());
		size_t expected_stride = 1;
		bool contig_if_nonempty = true;
		for (int64_t i = dim - 1; i >= 0; i--)
		{
			if (sizes[i] == 0)
			{
				return true;
			}
			if (contig_if_nonempty)
			{
				if (sizes[i] != 1 && strides[i] != expected_stride)
				{
					contig_if_nonempty = false;
				}
				expected_stride *= sizes[i];
			}
		}
		return contig_if_nonempty;
	}
	
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
			Shape const &s, const tart::DType& d, bool is_train) :
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        offset_(offset),
        capacity_(s.total_size()*d.size()),
        full_capacity_(capacity_ + offset * d.size())
    {
        buffer_ = buffer;
    }
    
    Tensor::Tensor(
			const tart::buffer_ptr& buffer,
			uint64_t offset,
			const Shape& sh, // shape
			const Shape& st, // strides
			const tart::DType& dt,
			bool trainable):
		specs_(std::make_shared<TensorSpecs>(sh, st, dt, trainable)),
		host_(std::make_shared<HostMem>()),
		offset_(offset),
		capacity_(sh.total_size()*dt.size()), // todo: see if this actually needs to be different because of stride weirdness
		full_capacity_(capacity_ + offset * dt.size())
	{
		buffer_ = buffer;
	}
    
    Tensor::Tensor(const tart::device_ptr& device, Shape const &s,const tart::DType& d,bool is_train):
        specs_(new TensorSpecs(s,d,is_train)),
		host_(new Tensor::HostMem()),
        offset_(0),
        capacity_(s.total_size()*d.size()),
        full_capacity_(capacity_)
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
		buffer_ = device->allocateBuffer(size);
		dev_ = device;
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

    void Tensor::to_device(void *p,bool sync)
    {
		buffer_->copyIn(p, memory_size(), offset_ * dtype().size());
    }

    void Tensor::to_device(bool sync)
    {
		buffer_->copyIn(host_data(), memory_size(), offset_ * dtype().size());
    }
    void Tensor::to_host(void *p,bool sync)
    {
        {
			buffer_->copyOut(p, memory_size(), offset_ * dtype().size());
		}
    }
    void Tensor::to_host(bool sync)
    {
		// all buffer copies in tart are sync, sorry :c
		buffer_->copyOut(host_data(), memory_size(), offset_ * dtype().size());
    }

    Tensor Tensor::sub_tensor(size_t offset,Shape const &s,const tart::DType& d,bool trainable) const
    {
        size_t offset_bytes = offset * dtype().size();
        DLPRIM_CHECK(shape().total_size()*dtype().size() >= s.total_size() * d.size());
        DLPRIM_CHECK((offset_ * dtype().size() + offset_bytes) % d.size() == 0);
        Tensor r;
        r.specs_.reset(new TensorSpecs(s,d,trainable));
        r.host_ = host_;
        r.buffer_ = buffer_;
        r.capacity_ = r.memory_size();
        r.full_capacity_ = full_capacity_;
        r.offset_ = (offset_ * dtype().size()  + offset_bytes) / d.size();
        return r;
    }

    void *Tensor::host_data()
    {
		// for now, though I am not happy about this
		if(host_->mHostMem.size() == 0)
		{
			host_->alloc(full_capacity_);
		}
		return (char*)(host_->mHostMem.data()) + offset_ * dtype().size();
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
