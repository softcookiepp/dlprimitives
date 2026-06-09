#pragma once
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>

#if VULKAN_API
namespace dlprim
{
	
namespace gpu
{

void hvol2col();

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
    const uint32_t data_col_offset); // new arg


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
    const DataType dtype); // also new arg

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
		const DataType accT);

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
		const DataType dtype);

} // namespace gpu
	
} // namespace dlprim
#endif // VULKAN_API
