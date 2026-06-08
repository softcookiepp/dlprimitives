#version 450
#include "../common/defs.glsl"

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer data_im_buf { dtype data_im[]; };
	layout(binding = 1, std430) writeonly buffer data_col_buf { dtype data_col[]; };
#endif

layout(push_constant, std430) uniform im2col_kernel
{
	uint n;
	#if USE_BDA
		const dtype* data_im;
	#endif
	uint data_im_offset; // new arg
	uint height;
	uint width;
	uint kernel_height;
	uint kernel_width;
	uint pad_height;
	uint pad_width;
	uint stride_height;
	uint stride_width;
	uint dilation_height;
	uint dilation_width;
	uint height_col;
	uint width_col;
	#if USE_BDA
		dtype* data_col;
	#endif
	uint data_col_offset;
};


// borrowed from pytorch
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type) \
	uint _i_n_d_e_x = (gl_WorkGroupID.x * gl_WorkGroupSize.x) + gl_LocalInvocationID.x; \
	for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+= gl_WorkGroupSize.x * gl_NumWorkGroups.x, i=_i_n_d_e_x)

void main()
{
	CUDA_KERNEL_LOOP_TYPE(index, n, uint)
	{
		uint w_out = index % width_col;

		uint idx = index / width_col;

		uint h_out = idx % height_col;
		uint channel_in = idx / height_col;
		uint channel_out = channel_in * kernel_height * kernel_width;
		uint h_in = h_out * stride_height - pad_height;
		uint w_in = w_out * stride_width - pad_width;

		uint col_offset = (channel_out * height_col + h_out) * width_col + w_out + data_col_offset;
		uint im_offset = (channel_in * height + h_in) * width + w_in + data_im_offset;

		for (uint i = 0; i < kernel_height; ++i)
		{
			for (uint j = 0; j < kernel_width; ++j)
			{
				uint h = h_in + i * dilation_height;
				uint w = w_in + j * dilation_width;
				data_col[col_offset] = (h >= 0 && w >= 0 && h < height && w < width)
						? data_im[i * dilation_height * width + j * dilation_width + im_offset]
						: dtype(0.0);
				col_offset += (height_col * width_col);
			}
		}
	}
}
