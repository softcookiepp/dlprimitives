
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
// better to have all constants spaced together I guess
layout(constant_id = 3) const uint SMEM_SIZE = 1;
#include "softmax_torch_common.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) writeonly buffer outp_buffer { dtype outp[]; };
	layout(binding = 1, std430) readonly buffer inp_buffer { dtype inp[]; };
#endif

layout(push_constant, std430) uniform cunn_SpatialSoftMaxForward
{
	#if USE_BDA
		dtype *outp;
	#endif
	uint outp_offset;
	#if USE_BDA
		const dtype *inp;
	#endif
	uint inp_offset;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

shared dtype sdata[SMEM_SIZE];

void do_softmax_forward()
{
	const uint outer_stride = inner_size * dim_size;
	const uint dim_stride = inner_size;

	for (uint outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x)
	{
		const uint outer_offset = outer_index * outer_stride;
		for (uint inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y)
		{
			const uint data_offset = outer_offset + inner_index;
			// TODO: make this determined by a specialization constant instead; they are for this exact purpose!
			////////////////////////////////////////////////////////////
			// These two blocks are really equivalent, but specializing on
			// blockDim.x == 1 makes the kernel faster when it's unused.
			// I didn't want to thread an extra template parameter, and nvcc
			// seems to be smart enough to hoist the if outside of the loops.
			////////////////////////////////////////////////////////////
#if 1
			if (blockDim.x > 1)
			{
				dtype max_input = -DTYPE_MAX;
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x) {
					const dtype value = dtype(inp[data_offset + d * dim_stride]);
					max_input = max(max_input, value);
				}
				spatialBlockReduceX(max_input, max, sdata, 0, max_input);

				dtype sum = 0;
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x)
					sum += exp(dtype(inp[data_offset + d * dim_stride]) - max_input);
				spatialBlockReduceX(sum, add_fn, sdata, 0, sum);

				Epilogue epilogue = Epilogue(max_input, sum);
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x)
				{
					dtype s;
					dtype si = inp[data_offset + d * dim_stride + inp_offset];
					do_epilogue(epilogue, s, si);
					outp[data_offset + d * dim_stride + outp_offset] = s;
				}
			}
			else
#endif
			{
				dtype max_input = -DTYPE_MAX;
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x)
				{
					const dtype value = dtype(inp[data_offset + d * dim_stride + inp_offset]);
					max_input = max(max_input, value);
				}
				dtype sum = 0;
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x)
					sum += exp(dtype(inp[data_offset + d * dim_stride + inp_offset]) - max_input);
				Epilogue epilogue = Epilogue(max_input, sum);
				
				for (uint d = threadIdx.x; d < dim_size; d += blockDim.x)
				{
#if 1
					do_epilogue(epilogue, outp[data_offset + d * dim_stride + outp_offset], inp[data_offset + d * dim_stride + inp_offset]);
#else
					dtype s;
					dtype si = inp[data_offset + d * dim_stride + inp_offset];
					do_epilogue(epilogue, s, si);
					outp[data_offset + d * dim_stride + outp_offset] = s;
#endif
				}
			}
		}
	}
}
