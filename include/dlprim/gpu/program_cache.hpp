///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/context.hpp>
#include <mutex>
#include <map>
#include <unordered_map>

namespace dlprim {
    namespace gpu {

        inline int round_up(int x,int y)
        {
            return (x+(y-1))/y*y;
        }

        inline
#if VULKAN_API
			std::vector<uint32_t>
#else
			cl::NDRange
#endif
			round_range(int x,
#if VULKAN_API
				std::vector<uint32_t> const &l)
#else
				cl::NDRange const &l)
#endif
        {
#if VULKAN_API
			std::vector<uint32_t> rounded(1);
			x = round_up(x, l[0]);
			rounded[0] = x;
			return rounded;
#else
            size_t const *size = l;
            x=round_up(x,size[0]);
            return cl::NDRange(x);
#endif
        }
        
#if VULKAN_API
		inline std::vector<uint32_t>
#else
        inline cl::NDRange
#endif
			round_range(int x,int y, 
#if VULKAN_API
				std::vector<uint32_t> const &l)
#else
				cl::NDRange const &l)
#endif
        {
#if VULKAN_API
			std::vector<uint32_t> rounded(2);
			rounded[0] = round_up(x, l[0]);
			rounded[1] = round_up(y, l[1]);
			return rounded;
#else
            size_t const *size = l;
            x=round_up(x,size[0]);
            y=round_up(y,size[1]);
            return cl::NDRange(x,y);
#endif
        }
#if VULKAN_API
		inline std::vector<uint32_t>
#else
        inline cl::NDRange
#endif
			round_range(int x,int y,int z,
#if VULKAN_API
				std::vector<uint32_t> const &l)
#else
				cl::NDRange const &l)
#endif
        {
#if VULKAN_API
			std::vector<uint32_t> rounded(3);
			rounded[0] = round_up(x, l[0]);
			rounded[1] = round_up(y, l[1]);
			rounded[2] = round_up(z, l[2]);
			return rounded;
#else
            size_t const *size = l;
            x=round_up(x,size[0]);
            y=round_up(y,size[1]);
            z=round_up(z,size[2]);
            return cl::NDRange(x,y,z);
#endif
        }
#if VULKAN_API
		extern std::map<std::string, std::map<std::string,std::string>> kernel_sources;
#else
        extern std::map<std::string,std::string> kernel_sources;
#endif

        struct Parameter {
            Parameter(std::string const &n,int v):
                name(n), value(std::to_string(v))
            {
            }
            Parameter(std::string const &n,std::string const &v):
                name(n), value(v)
            {
            }

            std::string name;
            std::string value;
        };

        class Cache {
        public:
            static Cache &instance();
            
            static void fill_params(std::vector<Parameter> &)
            {
            }

            template<typename Val,typename... Args>
            static void fill_params(std::vector<Parameter> &p,std::string const &n,Val v,Args... args)
            {
                p.push_back(Parameter(n,v));
                fill_params(p,args...);
            }

            template<typename Val,typename... Args>
#if VULKAN_API
			tart::program_ptr
#else
            cl::Program const &
#endif
				get_program(Context  &ctx,std::string const &source,std::string const &n1,Val const &v1,Args...args)
            {
                std::vector<Parameter> p;
                fill_params(p,n1,v1,args...);
                return get_program(ctx,source,p);
            }
#if VULKAN_API
			tart::program_ptr
#else
            cl::Program const &
#endif
				get_program(Context  &ctx,std::string const &source)
            {
                std::vector<Parameter> p;
                return get_program(ctx,source,p);
            }
#if VULKAN_API
			tart::program_ptr
#else
            cl::Program const &
#endif
				get_program(Context  &ctx,std::string const &source,std::vector<Parameter> const &params);

            template<typename Val,typename... Args>
#if VULKAN_API
			static tart::program_ptr
#else
            static cl::Program
#endif
				build_program(Context  &ctx,std::string const &source,std::string const &n1,Val const &v1,Args...args)
            {
                std::vector<Parameter> p;
                fill_params(p,n1,v1,args...);
                return build_program(ctx,source,p);
            }
            
#if VULKAN_API
			static tart::program_ptr
#else
            static cl::Program
#endif
				build_program(Context  &ctx,std::string const &source)
            {
                std::vector<Parameter> p;
                return build_program(ctx,source,p);
            }
#if VULKAN_API
			static tart::program_ptr
#else
            static cl::Program
#endif
				build_program(Context &ctx,std::string const &source,std::vector<Parameter> const &params);
        private:
            static std::string make_key(Context &ctx,std::string const &src,std::vector<Parameter> const &params);
            std::unordered_map<std::string,
#if VULKAN_API
				tart::program_ptr
#else
				cl::Program
#endif
				> cache_;
            std::mutex mutex_;
        };
    }
}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

