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
		tart::device_ptr dev = ec.queue_;
		const tart::device_ptr& ctx = dev;

        device_ = dev;
    }

    Context::Context(
		tart::device_ptr d
		) : 
		device_(d)
    {
    }
    
    Context::Context(std::string const &dev_id)
    {
        std::istringstream ss(dev_id);
        int p=-1,d=-1;
        char demim = 0;
        ss >>p >> demim >> d;
        if(!ss || demim != ':' || !ss.eof()) {
            throw ValidationError("Invalid device identification expecting `platform_no:device_no`");
        }
        select_device(d);
    }
    
    Context::Context(int device)
    {
        select_device(device);
    }

    std::string Context::name() const
    {
		return device_->getMetadata().name();
    }

    void Context::select_device(int d)
    {
		tart::Instance& instance = sInstance;
		if (d >= instance.getNumDevices() )
			throw ValidationError("No such device : " + std::to_string(d));
		device_ = instance.getDevice(d);
    }
}

