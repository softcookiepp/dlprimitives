///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/opencl_include.hpp>
#include <dlprim/definitions.hpp>
#include <chrono>
#include <stack>
#include <map>
#include <memory>

namespace dlprim {

class Context;

///
/// This class is used to pass cl::Events that the kernel should wait for and/or signal event completion
///
/// It is also used to pass cl::CommandQueue over API
///
/// Use it as following:
///
/// \code
/// void do_stuff(...,ExecutionContext const &e)
/// {
///  ...
//// e.queue().enqueueNDRangeKernel(kernel,nd1,nd2,nd3,
///                                   e.events(), // <- events to wait, nullptr of none
///                                   e.event("do_stuff")); //<- event to signal,
///                                                         // if profiling is used will be recorderd
///                                                         // under this name
/// }
/// \endcode
///
///
/// If you need to run several kernels use generate_series_context(#,total)
///
/// For example:
///
/// \code
///   run_data_preparation(e.generate_series_context(0,3)); // waits for events if needed
///   run_processing(e.generate_series_context(1,3)); // no events waited,signaled
///   run_reduce(e.generate_series_context(2,3)); // signals completion event if needed
/// \endcode
class ExecutionContext {
public:
    /// default constructor - can be used for CPU context
    ExecutionContext() {}

    ///
    /// Create context from cl::CommandQueue, note no events will be waited/signaled
    ///
    ExecutionContext(tart::device_ptr dev) :
		queue_(dev)
    {
    }

    ExecutionContext(ExecutionContext const &) = default;
    ExecutionContext &operator=(ExecutionContext const &) = default;

    ///
    /// Create contexts for multiple enqueues. 
    ///
    /// The idea is simple if we have events to signal and wait for and multiple
    /// kernels to execute, the first execution id == 0 should provide list of events
    /// to wait if id == total - 1, give event to signal
    ///
    ExecutionContext generate_series_context(size_t id,size_t total) const
    {
        ExecutionContext ctx = generate_series_context_impl(id,total);
        return ctx;
    }

    void finish()
    {
        if(queue_)
            queue_->sync();
    }

    
    ///
    /// Get the command queue. Never call it in non-OpenCL context
    ///
	tart::device_ptr queue() const
    {
        DLPRIM_CHECK(queue_);
        return queue_;
    }


private:
    ExecutionContext generate_series_context_impl(size_t id, size_t total) const
    {
		return ExecutionContext(queue());
    }


	tart::device_ptr queue_;
    friend class Context;
};


///
/// This is main object that represent the pair of OpenCL platform and device
/// all other objects use it.
///
/// It can be CPU context - meaning that it represents no OpenCL platform/device/context
///
///
class Context {
	static tart::Instance sInstance;
public:
    ///
    /// Create new context from textual ID. It can be "P:D"
    /// were P is integer representing platform and D is device number on this platform
    /// starting from 0, for example "0:1" is second device on 1st platform.
    ///
    Context(std::string const &dev_id);
    ///
    /// Create context from device number
    ///
    Context(int device = 0);
    ///
    /// Create the object from OpenCL context, platform and device..
    ///
    Context(
		tart::device_ptr d
	);

    /// 
    /// Create the object from queue
    ///
    Context(ExecutionContext const &ec);

    Context(Context const &) = default;
    Context &operator=(Context const &) = default;
    Context(Context &&) = default;
    Context &operator=(Context &&) = default;
    ~Context() {}

    ///
    /// Human readable name for the context, for example:
    /// "GeForce GTX 960 on NVIDIA CUDA"
    ///
    std::string name() const;

	// get tart instance
	static tart::Instance& getInstance() { return sInstance; }
    
    /// Get OpenCL device object
	tart::device_ptr
		device()
    {
        return device_;
    }

    /// Generate ExecutionContext (queue + events)
    ExecutionContext make_execution_context(uint32_t props=0)
    {
		return ExecutionContext(device_);
    }

private:
    void select_device(int d);
    tart::device_ptr device_;
    std::map<std::string,bool> ext_cache_;
};



class ExecGuard {
public:
    ExecGuard(ExecGuard const &) = delete;
    void operator=(ExecGuard const &) = delete;
    ExecGuard(ExecutionContext const &ctx,char const *name) : ctx_(&ctx)
    {
        // this no longer does anything. sorry friends
    }
    ~ExecGuard()
    {
    }
private:
    ExecutionContext const *ctx_;
};


} // namespace

