#if VULKAN_API

#include <dlprim/gpu/im2col.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim
{
	
namespace gpu
{

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
	const uint32_t data_col_offset, // new arg
	const DataType dtype)
{
	Context ctx(e);
	tart::program_ptr prg = Cache::instance().get_program(ctx, "im2col_torch", "dtype", data_type_to_opencl_type(dtype));
	tart::kernel_ptr im2colKernel = prg->getKernel("im2col");
	
	const uint32_t num_kernels = channels * height_col * width_col;
	const uint32_t threads_per_block = 1024;
	
	uint32_t block_num = (num_kernels - 1) / threads_per_block + 1;
	
	size_t p = 0;
	
	im2colKernel->setArg(p++, num_kernels);
	im2colKernel->setArg(p++, data_im);
	im2colKernel->setArg(p++, data_im_offset);
	im2colKernel->setArg(p++, height);
	im2colKernel->setArg(p++, width);
	im2colKernel->setArg(p++, kernel_height);
	im2colKernel->setArg(p++, kernel_width);
	im2colKernel->setArg(p++, pad_height);
	im2colKernel->setArg(p++, pad_width);
	im2colKernel->setArg(p++, stride_height);
	im2colKernel->setArg(p++, stride_width);
	im2colKernel->setArg(p++, dilation_height);
	im2colKernel->setArg(p++, dilation_width);
	im2colKernel->setArg(p++, height_col);
	im2colKernel->setArg(p++, width_col);

	im2colKernel->setArg(p++, data_col);
	im2colKernel->setArg(p++, data_col_offset);
	
	im2colKernel->run({block_num, 1, 1}, {});
}

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
	TypeTraits<T> traits;
	
	im2col(e,
		data_im,
		data_im_offset,
		channels,
		height,
		width,
		height_col,
		width_col,
		kernel_height,
		kernel_width,
		pad_height,
		pad_width,
		stride_height,
		stride_width,
		dilation_height,
		dilation_width,
		data_col,
		data_col_offset,
		traits.data_type);
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
