///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/activation.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    void activation_forward(Tensor &x,Tensor &y,StandardActivations activation, ExecutionContext const &ec,
		const tart::command_sequence_ptr& sequence)
    {
        Context ctx(ec);
		tart::program_ptr prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        tart::kernel_ptr k = prog->getKernel("activation");
        int p=0;
		uint32_t size = x.shape().total_size();
        k->setArg(p++,size);
        x.set_arg(k,p);
        y.set_arg(k,p);
		std::vector<uint32_t> wg({256, 1, 1});
		std::vector<uint32_t> gr = gpu::round_range(size, wg);
		for (size_t i = 0; i < gr.size(); i += 1)
			gr[i] = gr[i] / wg[i];
		gr.resize(3, 1);
		enqueue_or_record(k, gr, wg, sequence);
    }
    void activation_backward(Tensor &dx,Tensor &dy,Tensor &y,StandardActivations activation,float beta,ExecutionContext const &ec, const tart::command_sequence_ptr& sequence)
    {
        Context ctx(ec);
		tart::program_ptr const &prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                    "ACTIVATION",int(activation));
        tart::kernel_ptr k = prog->getKernel("activation_diff");
        
        int p=0;
        uint32_t size = y.shape().total_size();
        k->setArg(p++,size);
        y.set_arg(k,p);
        dy.set_arg(k,p);
        dx.set_arg(k,p);
        k->setArg(p++,beta);

        std::vector<uint32_t> wg({256, 1, 1});
        std::vector<uint32_t> gr=gpu::round_range(size,wg);
        gr[0] = gr[0]/wg[0];
        gr.resize(3, 1);
        enqueue_or_record(k, gr, wg, sequence);
    }

} // core
} // dlprim
