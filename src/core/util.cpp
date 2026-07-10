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
#if VULKAN_API
                        tart::buffer_ptr& src, uint32_t src_offset, Shape src_strides,
                        tart::buffer_ptr& dst, uint32_t dst_offset, Shape dst_strides,
#else
                        cl::Buffer const &src,cl_ulong src_offset,Shape src_strides,
                        cl::Buffer const &dst,cl_ulong dst_offset,Shape dst_strides,
#endif
                        DataType dtype_src,
                        DataType dtype_dst,
                        ExecutionContext const &q)
    {
        DLPRIM_CHECK(shape.size() == src_strides.size());
        DLPRIM_CHECK(shape.size() == dst_strides.size());
        int dims = shape.size();
        Context ctx(q);
        bool use_io_type = dtype_src == dtype_dst;
#if VULKAN_API
		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx,"copy_strided",
                                "dtype_src",data_type_to_opencl_type(dtype_src,use_io_type),
                                "dtype_tgt",data_type_to_opencl_type(dtype_dst,use_io_type),
                                "DIMS",dims);
        std::vector<uint32_t> range;
        switch(dims) {
        case 1: range = {shape[0]}; break;
        case 2: range = {shape[1],shape[0]}; break;
        case 3: range = {shape[2],shape[1],shape[0]}; break;
        case 4: range = {shape[3]*shape[2],shape[1],shape[0]}; break;
        case 5: range = {shape[4]*shape[3],shape[2]*shape[1],shape[0]}; break;
        case 6: range = {shape[5]*shape[4],shape[3]*shape[2],shape[1]*shape[0]}; break;
        case 7: range = {shape[6]*shape[5]*shape[4],shape[3]*shape[2],shape[1]*shape[0]}; break;
        case 8: range = {shape[7]*shape[6]*shape[5],shape[4]*shape[3]*shape[2],shape[1]*shape[0]}; break;
        default:
            throw NotImplementedError("Invalid dimentsions count for strided copy " + std::to_string(dims));
        }
        tart::kernel_ptr k = prog->getKernel("copy");
        int p=0;
        for(int i=0;i<dims;i++) {
            k->setArg(p++, uint32_t(shape[i]));
            k->setArg(p++, uint32_t(src_strides[i]));
            k->setArg(p++, uint32_t(dst_strides[i]));
        }
        k->setArg(p++,src);
        k->setArg(p++,src_offset);
        k->setArg(p++,dst);
        k->setArg(p++,dst_offset);
        std::vector<uint32_t> local;
        k->enqueue(range, local);
#else
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"copy_strided",
                                "dtype_src",data_type_to_opencl_type(dtype_src,use_io_type),
                                "dtype_tgt",data_type_to_opencl_type(dtype_dst,use_io_type),
                                "DIMS",dims);
        cl::NDRange range;
        switch(dims) {
        case 1: range = cl::NDRange(shape[0]); break;
        case 2: range = cl::NDRange(shape[1],shape[0]); break;
        case 3: range = cl::NDRange(shape[2],shape[1],shape[0]); break;
        case 4: range = cl::NDRange(shape[3]*shape[2],shape[1],shape[0]); break;
        case 5: range = cl::NDRange(shape[4]*shape[3],shape[2]*shape[1],shape[0]); break;
        case 6: range = cl::NDRange(shape[5]*shape[4],shape[3]*shape[2],shape[1]*shape[0]); break;
        case 7: range = cl::NDRange(shape[6]*shape[5]*shape[4],shape[3]*shape[2],shape[1]*shape[0]); break;
        case 8: range = cl::NDRange(shape[7]*shape[6]*shape[5],shape[4]*shape[3]*shape[2],shape[1]*shape[0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for strided copy " + std::to_string(dims));
        }
        cl::Kernel k(prog,"copy");
        int p=0;
        for(int i=0;i<dims;i++) {
            k.setArg(p++,cl_ulong(shape[i]));
            k.setArg(p++,cl_ulong(src_strides[i]));
            k.setArg(p++,cl_ulong(dst_strides[i]));
        }
        k.setArg(p++,src);
        k.setArg(p++,src_offset);
        k.setArg(p++,dst);
        k.setArg(p++,dst_offset);
        q.queue().enqueueNDRangeKernel(k,cl::NullRange,range,cl::NullRange,q.events(),q.event("copy_strided"));
#endif
    }
}
}

