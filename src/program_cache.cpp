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
namespace gpu {


#ifdef DEBUG_CACHE_TIMES
class TimeWriter {
public:
    decltype(std::chrono::high_resolution_clock::now()) start;
    std::string name;
    TimeWriter(std::string const &n) : name(n)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    ~TimeWriter()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        auto passed = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
        std::cout << "Kernel " << name << " " << passed * 1e3 << " ms" << std::endl;
    }
};
#endif

Cache &Cache::instance()
{
    static Cache c;
    return c;
}

tart::program_ptr Cache::get_program(Context &ctx,std::string const &source,std::vector<Parameter> const &params)
{
	std::string key = make_key(ctx.device(), source, params);
    std::unique_lock<std::mutex> g(mutex_);
    auto p = cache_.find(key);
    if(p == cache_.end())
    {
		auto prg = build_program(ctx.device(), source, params);
        cache_[key]=prg;
    }
    return cache_[key];

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
				throw std::runtime_error("dtype cannot be empty");
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

std::string Cache::make_key(const tart::device_ptr& device,std::string const &src,std::vector<Parameter> const &params)
{
	std::uintptr_t ctx_ptr = (std::uintptr_t)(device.get());
    std::ostringstream ss;
    ss << "prg:" << ctx_ptr <<  "@" << src <<  "/?";
    for(size_t i=0;i<params.size();i++) {
        if(i > 0)
            ss << '&';
        ss << params[i].name << '=' << params[i].value;
    }
    return ss.str();
}


}
}
