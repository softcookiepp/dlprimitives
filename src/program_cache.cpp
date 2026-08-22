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
		if (startChar == '$')
		{
			prepend << "\n" << params[i].value << "\n";
		}
        else if(startChar == '#') {
            prepend << "#define " << params[i].name.c_str() + 1 << " " << params[i].value << "\n";
            combine=true;
        }
        else
        {
			std::stringstream optSS;
			optSS << "-D" << params[i].name <<"=" <<params[i].value;
			// std::cout << "OPTION: " << optSS.str() << "\n";
            options.push_back(optSS.str());
            if (params[i].name == "dtype" && params[i].value == "")
            {
				//throw std::runtime_error("dtype cannot be empty");
				params[i].value = "float";
			}
        }
    }
    
	tart::DeviceMetadata meta = device->getMetadata();
    // storage extensions
    if (meta.half_ || meta.short_) prepend << "#extension GL_EXT_shader_16bit_storage : require\n";
    if(meta.char_) prepend << "#extension GL_EXT_shader_8bit_storage : require\n";
    
    // arithmetic types
    if(meta.double_) prepend << "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
    if(meta.half_) prepend << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    if(meta.long_) prepend << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
    if(meta.short_) prepend << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    if(meta.char_) prepend << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    
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
