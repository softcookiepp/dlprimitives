
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

	for (uint outer_index = get_group_id(0); outer_index < outer_size; outer_index += get_num_groups(0))
	{
		const uint outer_offset = outer_index * outer_stride;
		// wait isn't that just the global id? I am going to leave it as is out of paranoia
		for (uint inner_index = get_group_id(1) * get_local_size(1) + get_local_id(1); inner_index < inner_size; inner_index += get_local_size(1) * get_num_groups(1))
		{
			const uint data_offset = outer_offset + inner_index;
			dtype max_input = DTYPE_MIN;
			if (get_local_size(0) > 1)
			{
				
				for (uint d = get_local_id(0); d < dim_size; d += get_local_size(0)) {
					const dtype value = dtype(inp[data_offset + d * dim_stride + inp_offset]);
					max_input = max(max_input, value);
				}
				spatialBlockReduceX(max_input, max, sdata, 0, max_input);
				//max_input = spatialBlockReduceX(sdata,max_input);

				dtype sum = 0;
				for (uint d = get_local_id(0); d < dim_size; d += get_local_size(0))
					sum += exp(dtype(inp[data_offset + d * dim_stride + inp_offset])
								 - max_input);
				//sum = spatialBlockReduceX<dtype, Add>(sdata, sum);
				spatialBlockReduceX(sum, add_fn, sdata, 0, sum);

				Epilogue epilogue = Epilogue(max_input, sum);
				for (uint d = get_global_id(0); d < dim_size; d += get_local_size(0))
					do_epilogue(epilogue, outp[data_offset + d * dim_stride + outp_offset], inp[data_offset + d * dim_stride + inp_offset]);
			}
			else
			{
				for (uint d = get_global_id(0); d < dim_size; d += get_local_size(0)) {
					const dtype value = dtype(inp[data_offset + d * dim_stride + inp_offset]);
					max_input = max(max_input, value);
				}
				dtype sum = 0;
				for (uint d = get_global_id(0); d < dim_size; d += get_local_size(0))
					sum += exp(dtype(inp[data_offset + d * dim_stride + inp_offset]) - max_input);
				Epilogue epilogue = Epilogue(max_input, sum);
				for (uint d = get_global_id(0); d < dim_size; d += get_local_size(0))
					do_epilogue(epilogue, outp[data_offset + d * dim_stride + outp_offset], inp[data_offset + d * dim_stride + inp_offset]);
			}
		}
	}
}
