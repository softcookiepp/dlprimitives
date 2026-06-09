#version 450

#include "../common/defs.glsl"

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer data_col_buf { dtype data_col[]; };
	layout(binding = 1, std430) writeonly buffer data_im_buf { dtype data_im[]; };
#endif

layout(push_constant, std430) uniform col2im_kernel
{
	uint n;
	#if USE_BDA
		const dtype* data_col;
	#endif
	uint data_col_offset;
	uint height;
	uint width;
	uint kernel_h;
	uint kernel_w;
	uint pad_height;
	uint pad_width;
	uint stride_height;
	uint stride_width;
	uint dilation_height;
	uint dilation_width;
	uint height_col;
	uint width_col;
	#if USE_BDA
		dtype* data_im
	#endif
	uint data_im_offset;
} args; // to prevent namespace pollution

#include "col2im_device.glsl"

void main()
{
	CUDA_KERNEL_LOOP(index, args.n)
	{
		col2im_device(
			index,
			#if USE_BDA
				data_col,
			#endif
			args.data_col_offset,
			args.height,
			args.width,
			args.kernel_h,
			args.kernel_w,
			args.pad_height,
			args.pad_width,
			args.stride_height,
			args.stride_width,
			args.dilation_height,
			args.dilation_width,
			args.height_col,
			args.width_col,
			#if USE_BDA
				data_im,
			#endif
			args.data_im_offset);
	}
}
