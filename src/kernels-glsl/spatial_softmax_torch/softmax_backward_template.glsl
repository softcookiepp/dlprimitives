
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
layout(constant_id = 3) const uint SMEM_SIZE = 1;

// define this before including so that it doesn't do the forward one
#ifndef SOFTMAX_EPILOGUE_TYPE
	#define SOFTMAX_EPILOGUE_TYPE SOFTMAX_BACKWARD_EPILOGUE
#endif

#include "softmax_torch_common.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer gradInputBuf { dtype gradInput[]; };
	layout(binding = 1, std430) buffer outputBuf { dtype output_[]; };
	layout(binding = 2, std430) buffer gradOutputBuf { dtype gradOutput[]; };
#endif

shared dtype sdata[SMEM_SIZE];

layout(push_constant, std430) uniform cunn_SpatialSoftMaxBackward
{
	#if USE_BDA
		dtype *gradInput;
	#endif
	uint gradInputOffset;
	#if USE_BDA
		const dtype *output_;
	#endif
	uint output_offset;
	#if USE_BDA
		const dtype *gradOutput;
	#endif
	uint gradOutputOffset;
	uint outer_size; uint dim_size; uint inner_size;
};

void do_softmax_backward()
{
	const uint32_t outer_stride = inner_size * dim_size;
	const uint32_t dim_stride = inner_size;

	for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x)
	{
		const uint32_t outer_offset = outer_index * outer_stride;
		for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y)
		{
			const uint32_t data_offset = outer_offset + inner_index;
			// See the comment in forward kernel
			if (blockDim.x > 1)
			{
				accscalar_t sum = 0;
				for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
					sum += gradOutput[data_offset + d * dim_stride];
				//sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);
				spatialBlockReduceX(sum, add_fn, sdata, 0, sum)

				//Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
				Epilogue epilogue = Epilogue(sum);
				for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
				{
					gradInput[data_offset + d * dim_stride] =
						do_epilogue(epilogue, gradOutput[data_offset + d * dim_stride],
										output[data_offset + d * dim_stride]);
				}
			}
			else
			{
				accscalar_t sum = 0;
				for (uint32_t d = 0; d < dim_size; d++)
					sum += gradOutput[data_offset + d * dim_stride];

				Epilogue epilogue = Epilogue(sum);
				for (uint32_t d = 0; d < dim_size; d++) {
					gradInput[data_offset + d * dim_stride] =
						do_epilogue(epilogue, gradOutput[data_offset + d * dim_stride],
										output[data_offset + d * dim_stride]);
				}
			}
		}
	}
}
