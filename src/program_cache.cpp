///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include  <dlprim/gpu/program_cache.hpp>
#include <sstream>
#include <chrono>
#include <iostream>


//#define DEBUG_CACHE_TIMES 

namespace dlprim {
namespace gpu
{

Cache &Cache::instance()
{
    static Cache c;
    return c;
}

tart::program_ptr Cache::get_program(const tart::device_ptr& device, std::string const &source,std::vector<Parameter> const &params)
{
	return build_program(device, source, params);
}

tart::program_ptr Cache::build_program(const tart::device_ptr& device, std::string const &source,std::vector<Parameter> const &params)
{
	// std::cout << "	Getting program: " << source << std::endl;
	auto ks = kernel_sources.find(source);
    if(ks == kernel_sources.end())
        throw ValidationError("Unknow program source " + source);
	auto& entryPointMap = ks->second;
    //std::string const &source_text = ks->second;
    std::ostringstream prepend;
    bool combine = false;

    std::vector<std::string> options;
    for(size_t i=0;i<params.size();i++)
    {
		const char startChar = params[i].name.c_str()[0];
        {
			std::stringstream optSS;
			optSS << "-D" << params[i].name <<"=" <<params[i].value;
			// std::cout << "OPTION: " << optSS.str() << "\n";
            options.push_back(optSS.str());
        }
    }
    
	tart::DeviceMetadata meta = device->getMetadata();
    // storage extensions
    if (meta.half_ || meta.short_) options.push_back("-DENABLE_16BIT_STORAGE=1");
    if(meta.char_) options.push_back("-DENABLE_8BIT_STORAGE=1");
    
    // arithmetic types
    if(meta.double_) options.push_back("-DENABLE_FLOAT64_ARITHMETIC=1");
    if(meta.half_) options.push_back("-DENABLE_FLOAT16_ARITHMETIC=1");
    if(meta.long_) options.push_back("-DENABLE_INT64_ARITHMETIC=1");
    if(meta.short_) options.push_back("-DENABLE_INT16_ARITHMETIC=1");
    if(meta.char_)  options.push_back("-DENABLE_INT8_ARITHMETIC=1");
    
    if (meta.subgroupAdd) prepend << "#define USE_SUBGROUP_ARITHMETIC 1\n";
    
    if (prepend.str().size() > 0) combine = true;
    
	std::map<std::string, tart::shader_module_ptr> entryPointModules;
	for (auto& pair : entryPointMap)
	{
		const std::string src = "#version 450\n" + (combine ? prepend.str() + pair.second : pair.second);
		entryPointModules[pair.first] = device->compileGLSL(src, options, pair.first);
	}
	
    #ifdef DEBUG_CACHE_TIMES
    TimeWriter guard(source);
    #endif
    
	// this may be more complicated than I thought :c
	tart::program_ptr prg = device->createProgram(entryPointModules);
	return prg;
	// end
}

}
}
