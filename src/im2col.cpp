#include <dlprim/gpu/im2col.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>

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
	tart::program_ptr prg = PerDeviceProgramCache::instance().im2col(ctx.device(), dtype);
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
	
	im2colKernel->enqueue({block_num, 1, 1}, {});
}

void col2im(
		const ExecutionContext& e,
		const tart::buffer_ptr& data_col,
		const uint32_t data_col_offset,
		const uint32_t channels,
		const uint32_t height,
		const uint32_t width,
		const uint32_t height_col,
		const uint32_t width_col,
		const uint32_t patch_height,
		const uint32_t patch_width,
		const uint32_t pad_height,
		const uint32_t pad_width,
		const uint32_t stride_height,
		const uint32_t stride_width,
		const uint32_t dilation_height,
		const uint32_t dilation_width,
		const tart::buffer_ptr& data_im,
		const uint32_t data_im_offset,
		const DataType dtype,
		const DataType accT)
{
	uint32_t num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	// CUDA_NUM_THREADS = 1024
	uint32_t block_num = (num_kernels - 1) / 512 + 1;
	
	// get the kernel
	Context ctx(e);
	tart::program_ptr prg = PerDeviceProgramCache::instance().col2im(ctx.device(), dtype);
	tart::kernel_ptr col2im_kernel = prg->getKernel("col2im");

	size_t p = 0;
	col2im_kernel->setArg(p++, num_kernels);
	col2im_kernel->setArg(p++, data_col);
	col2im_kernel->setArg(p++, data_col_offset);
	col2im_kernel->setArg(p++, height);
	col2im_kernel->setArg(p++, width);
	col2im_kernel->setArg(p++, patch_height);
	col2im_kernel->setArg(p++, patch_width);
	col2im_kernel->setArg(p++, pad_height);
	col2im_kernel->setArg(p++, pad_width);
	col2im_kernel->setArg(p++, stride_height);
	col2im_kernel->setArg(p++, stride_width);
	col2im_kernel->setArg(p++, dilation_height);
	col2im_kernel->setArg(p++, dilation_width);
	col2im_kernel->setArg(p++, height_col);
	col2im_kernel->setArg(p++, width_col);
	col2im_kernel->setArg(p++, data_im);
	col2im_kernel->setArg(p++, data_im_offset);
	
	col2im_kernel->enqueue({block_num, 1, 1}, {});
}

void col2im_batched(
		const ExecutionContext& e,
		const tart::buffer_ptr& data_col,
		const uint32_t data_col_offset,
		const uint32_t col_batch_stride,
		const uint32_t nbatch,
		const uint32_t channels,
		const uint32_t height,
		const uint32_t width,
		const uint32_t height_col,
		const uint32_t width_col,
		const uint32_t patch_height,
		const uint32_t patch_width,
		const uint32_t pad_height,
		const uint32_t pad_width,
		const uint32_t stride_height,
		const uint32_t stride_width,
		const uint32_t dilation_height,
		const uint32_t dilation_width,
		const tart::buffer_ptr& data_im,
		const uint32_t data_im_offset,
		const uint32_t im_batch_stride,
		const DataType dtype)
{
	const uint32_t num_kernels = channels * height * width;
	const uint32_t output_numel = nbatch * num_kernels;
	if (output_numel == 0) {
		return;	// No work to do
	}
	
	// get the kernel
	Context ctx(e);
	tart::program_ptr prg = PerDeviceProgramCache::instance().col2im(ctx.device(), dtype);
	tart::kernel_ptr col2im_kernel = prg->getKernel("col2im_batched");
	
	size_t p = 0;
	col2im_kernel->setArg(p++, num_kernels);
	col2im_kernel->setArg(p++, data_col);
	col2im_kernel->setArg(p++, data_col_offset);
	col2im_kernel->setArg(p++, col_batch_stride);
	col2im_kernel->setArg(p++, nbatch);
	col2im_kernel->setArg(p++, height);
	col2im_kernel->setArg(p++, width);
	col2im_kernel->setArg(p++, patch_height);
	col2im_kernel->setArg(p++, patch_width);
	col2im_kernel->setArg(p++, pad_height);
	col2im_kernel->setArg(p++, pad_width);
	col2im_kernel->setArg(p++, stride_height);
	col2im_kernel->setArg(p++, stride_width);
	col2im_kernel->setArg(p++, dilation_height);
	col2im_kernel->setArg(p++, dilation_width);
	col2im_kernel->setArg(p++, height_col);
	col2im_kernel->setArg(p++, width_col);
	col2im_kernel->setArg(p++, data_im);
	col2im_kernel->setArg(p++, data_im_offset);
	col2im_kernel->setArg(p++, im_batch_stride);
	
	uint32_t block_num = (num_kernels - 1) / 512 + 1;
	
	col2im_kernel->enqueue({block_num, 1, 1}, {});
}


} // namespace gpu
	
} // namespace dlprim
