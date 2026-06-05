#if VULKAN_API

#include <dlprim/gpu/im2col.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim
{
	
namespace gpu
{

template <typename T>
void im2col(const ExecutionContext& e,
	const tart::buffer_ptr& data_im,
	const uint32_t data_im_offset, // new arg
	const uint32_t channels,
	const uint32_t height,
	const uint32_t width,
	const uint32_t height_col,
	const uint32_t width_col,
	const uint32_t kernel_height,
	const uint32_t kernel_width,
	const uint32_t pad_height,
	const uint32_t pad_width,
	const uint32_t stride_height,
	const uint32_t stride_width,
	const uint32_t dilation_height,
	const uint32_t dilation_width,
	const tart::buffer_ptr& data_col,
	const uint32_t data_col_offset) // new arg
{
	Context ctx(e);
	TypeTraits<T> traits;
	tart::program_ptr prg = Cache::instance().get_program(ctx, "im2col_torch", "dtype", data_type_to_opencl_type(traits.data_type));
	tart::kernel_ptr im2colKernel = prg->getKernel("im2col");
	
	im2colKernel->setArg(0, data_im);
	im2colKernel->setArg(1, data_im_offset);
	im2colKernel->setArg(2, channels);
	im2colKernel->setArg(3, height);
	im2colKernel->setArg(4, width);
	im2colKernel->setArg(5, height_col);
	im2colKernel->setArg(6, width_col);
	im2colKernel->setArg(7, kernel_height);
	im2colKernel->setArg(8, kernel_width);
	im2colKernel->setArg(9, pad_height);
	im2colKernel->setArg(10, pad_width);
	im2colKernel->setArg(11, stride_height);
	im2colKernel->setArg(12, stride_width);
	im2colKernel->setArg(13, dilation_height);
	im2colKernel->setArg(14, dilation_width);
	im2colKernel->setArg(15, data_col);
	im2colKernel->setArg(16, data_col_offset);
	
	const uint32_t num_kernels = channels * height_col * width_col;
	const uint32_t threads_per_block = 1024;
	
	uint32_t block_num = (num_kernels - 1) / threads_per_block + 1;
	im2colKernel->run({block_num, 1, 1}, {});
}

//template void im2col<half>();
template void im2col<float>(const ExecutionContext& e,
	const tart::buffer_ptr& data_im,
	const uint32_t data_im_offset, // new arg
	const uint32_t channels,
	const uint32_t height,
	const uint32_t width,
	const uint32_t height_col,
	const uint32_t width_col,
	const uint32_t kernel_height,
	const uint32_t kernel_width,
	const uint32_t pad_height,
	const uint32_t pad_width,
	const uint32_t stride_height,
	const uint32_t stride_width,
	const uint32_t dilation_height,
	const uint32_t dilation_width,
	const tart::buffer_ptr& data_col,
	const uint32_t data_col_offset);
//template void im2col<double>();


} // namespace gpu
	
} // namespace dlprim

#endif // VULKAN_API
