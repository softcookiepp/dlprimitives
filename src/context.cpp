///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include <dlprim/context.hpp>
#include <sstream>
#include <iostream>

namespace dlprim
{
	tart::Instance Context::sInstance;
    Context::Context(ExecutionContext const &ec)
    {
        if(!ec.queue_) {
            type_ = cpu;
            return;
        }
		tart::device_ptr dev = ec.queue_;
		const tart::device_ptr& ctx = dev;

		// just use the device as a placeholder, then even
		platform_ = dev;
        device_ = dev;
        context_ = ctx;
        // ??
        type_ = Context::ocl;
    }

    Context::Context(
		tart::device_ptr d
		) : 
		platform_(d),
		device_(d),
		context_(d),
        type_(Context::ocl)
    {
    }
    
    Context::Context(std::string const &dev_id)
    {
        if(dev_id == "cpu") {
            type_ = cpu;
            return;
        }
        std::istringstream ss(dev_id);
        int p=-1,d=-1;
        char demim = 0;
        ss >>p >> demim >> d;
        if(!ss || demim != ':' || !ss.eof()) {
            throw ValidationError("Invalid device identification expecting one of `cpu` or `paltform_no:device_no`");
        }
        type_ = ocl;
        select_opencl_device(p,d);
    }

    Context::Context(ContextType dt,int platform,int device) :
        type_(dt)
    {
        if(dt == cpu)
            return;
        select_opencl_device(platform,device);
    }

    bool Context::check_device_extension(std::string const &name)
    {
        bool res;
        auto p = ext_cache_.find(name);
        if(p == ext_cache_.end()) {
            res = device_extensions().find(name) != std::string::npos;
            ext_cache_[name] = res;
        }
        else {
            res = p->second;
        }
        return res;
    }

    int Context::estimated_core_count()
    {
		// not implemented in tart yet; will do later
		return 0;
    }

    std::string const &Context::device_extensions()
    {
		return ext_;
    }


    std::string Context::name() const
    {
        if(is_cpu_context())
            return "CPU";
		std::string name = "not implemented"; //device_->getMetadata().name();
		return name;
    }

    void Context::select_opencl_device(int p,int d)
    {
		tart::Instance& instance = sInstance;
		if (d >= instance.getNumDevices() )
			throw ValidationError("No such device : " + std::to_string(d));
		device_ = instance.getDevice(d);
		context_ = device_;
    }
}

