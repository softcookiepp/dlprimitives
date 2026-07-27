///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>

namespace dlprim {
namespace core {
    void copy_strided(  Shape shape,
                        tart::buffer_ptr& src, uint32_t src_offset, Shape src_strides,
                        tart::buffer_ptr& dst, uint32_t dst_offset, Shape dst_strides,
                        DataType dtype_src,
                        DataType dtype_dst,
                        ExecutionContext const &q)
    {
        DLPRIM_CHECK(shape.size() == src_strides.size());
        DLPRIM_CHECK(shape.size() == dst_strides.size());
        int dims = shape.size();
        Context ctx(q);
        bool use_io_type = dtype_src == dtype_dst;
		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx,"copy_strided",
                                "dtype_src",data_type_to_opencl_type(dtype_src,use_io_type),
                                "dtype_tgt",data_type_to_opencl_type(dtype_dst,use_io_type) );
        std::vector<uint32_t> range;
        switch(dims)
        {
			case 1: range = {shape[0]}; break;
			case 2: range = {shape[1], shape[0]}; break;
			case 3: range = {shape[2], shape[1], shape[0]}; break;
			case 4: range = {shape[3]*shape[2], shape[1], shape[0]}; break;
			case 5: range = {shape[4]*shape[3], shape[2]*shape[1], shape[0]}; break;
			case 6: range = {shape[5]*shape[4], shape[3]*shape[2], shape[1]*shape[0]}; break;
			case 7: range = {shape[6]*shape[5]*shape[4], shape[3]*shape[2], shape[1]*shape[0]}; break;
			case 8: range = {shape[7]*shape[6]*shape[5],shape[4]*shape[3]*shape[2], shape[1]*shape[0]}; break;
        default:
            throw NotImplementedError("Invalid dimentsions count for strided copy " + std::to_string(dims));
        }
        tart::kernel_ptr k = prog->getKernel("copy");
        int p=0;
        for(int i=0; i < 8; i++)
        {
			if (i < shape.size())
			{
				k->setArg(p++, uint32_t(shape[i]));
				k->setArg(p++, uint32_t(src_strides[i]));
				k->setArg(p++, uint32_t(dst_strides[i]));
			}
			else
			{
				// not all tensors will have 8 dimensions, leave data for remaining ones blank
				const uint32_t zero = 0;
				k->setArg(p++, zero);
				k->setArg(p++, zero);
				k->setArg(p++, zero);
			}
        }
        k->setArg(p++,src);
        k->setArg(p++,src_offset);
        k->setArg(p++,dst);
        k->setArg(p++,dst_offset);
        
        // Ensure GPU is properly saturated
        auto globalAndLocal = ctx.device()->chooseGlobalAndLocalSize(range);
        auto& local = globalAndLocal.second;
        
        std::vector<uint32_t> spec = {
			local[0], local[1], local[2], dims
		};
        k->enqueue(globalAndLocal.first, spec);
    }
}
}

