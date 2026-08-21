///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>
#include "test.hpp"

namespace dp = dlprim;

int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "Use paltform:device" << std::endl;
        return 1;
    }
    try {

        std::cout << "Basic context" << std::endl; 

        dp::Context ctx(argv[1]);
        std::cout << ctx.name() << std::endl;
        dp::Tensor a(ctx.device(), dp::Shape(10));

        dp::ExecutionContext q = ctx.make_execution_context();

        float *p = a.data<float>();

        for(unsigned i=0;i<a.shape()[0];i++)
            p[i] = -5.0 + i;
        a.to_device();
        tart::program_ptr prg = dp::gpu::PerDeviceProgramCache::instance().bias(ctx.device());
        tart::kernel_ptr k = prg->getKernel("activation_inplace");
        int pos=0;
        k->setArg(pos++,int(a.shape().total_size()));
        a.set_arg(k,pos);
        k->enqueue({a.shape().total_size()}, {1, 1, 1, uint32_t(dp::StandardActivations::relu)});
        a.to_host(false);
        for(unsigned i=0;i<a.shape()[0];i++) {
            TEST(p[i] == std::max(0.0,-5.0 + i));
        }
        std::cout << "Ok" << std::endl;
    }
    catch(std::exception const &e) {
        std::cerr <<"Failed:"<< e.what() << std::endl;
        return 1;
    }
    return 0;

}
