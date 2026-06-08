#pragma once
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>

#if VULKAN_API
namespace dlprim
{
	
namespace gpu
{

template <typename T>
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

} // namespace gpu
	
} // namespace dlprim
#endif // VULKAN_API
