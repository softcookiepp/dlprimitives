#if VULKAN_API // if no vulkan, no need for this crap

#include <dlprim/gpu/naive.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <iostream>
#include <clblast_vk.h>
#include <dlprim/core/pointwise.hpp>

namespace dlprim
{

namespace gpu
{

Tensor convolution_from_ggml_raw(const Tensor& input,
	const Tensor& weight,
	const std::optional<Tensor>& bias,
	const std::vector<int>& stride,
	const std::vector<int>& padding,
	const std::vector<int>& dilation,
	bool transposed,
	const std::vector<int>& output_padding,
	int64_t groups)
{
	if (input.shape().size() > 5)
		throw std::runtime_error("convolution not implemented for above 2d yet");
	
	// just do conv2d_mm by default
	// assume this:
#if 0
	bool transpose = transpose;

    vk_op_conv2d_push_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, std::move(p));
#endif
}

} // gpu

} // dlprim

#endif
