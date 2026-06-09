#version 450

#include "../common/defs.glsl"

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer data_col_buf { dtype data_col[]; };
	layout(binding = 1, std430) writeonly buffer data_im_buf { dtype data_im[]; };
#endif

layout(push_constant, std430) uniform col2im_batched_kernel
{
	uint n;
	#if USE_BDA
	const dt* data_col;
	#endif
	uint data_col_offset;
	uint col_batch_stride;
	uint nbatch;
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
		dt* data_im;
	#endif
	uint data_im_offset;
	uint im_batch_stride;
} args; // prevent namespace pollution, since col2im_device uses a lot of the same names :c

#include "col2im_device.glsl"

void main()
{
	const uint im_numel = args.n * args.nbatch;

	CUDA_KERNEL_LOOP_TYPE(index, im_numel, uint)
	{
		const uint ibatch = index / args.n;
		const uint slice_index = index % args.n;

		col2im_device(
				slice_index,
				#if USE_BDA
					data_col + args.ibatch * args.col_batch_stride,
				#endif
				args.data_col_offset + (ibatch * args.col_batch_stride),
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
					data_im + ibatch * im_batch_stride,
				#endif
				args.data_im_offset + (ibatch * args.im_batch_stride)
				);
	}
}
