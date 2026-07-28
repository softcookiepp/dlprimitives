#version 450
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

layout(local_size_x = 1, local_size_y_id = 0, local_size_z = 1) in;
layout(constant_id = 0) const uint WG_SIZE = 256;
layout(constant_id = 1) const uint ITEMS_PER_WI = 1;
layout(constant_id = 2) const uint POOL_MODE = 1;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { dtype inp[]; };
	layout(binding = 1, std430) writeonly buffer outp_buf { dtype outp[]; };
#endif

layout(push_constant, std430) uniform global_pooling
{
	uint items;
	uint over;
	float scale;
#if USE_BDA
	__global dtype *inp;
#endif
	uint data_offset;
#if USE_BDA
	__global dtype *outp;
#endif
	uint out_offset;
};

REDUCE_PREPARE(WG_SIZE,dtype);

void main()
{
	uint inp_offset_ = data_offset;
	uint outp_offset_ = out_offset;
	
	uint b = get_global_id(0);

	if(b >= items)
		return;

	uint c = get_global_id(1) * ITEMS_PER_WI;

	inp_offset_ += b * over;
	outp_offset_ += b;
	
	dtype val;
	
	if (POOL_MODE == 0)
	{
		val = -DTYPE_MAX;
		for(uint i=0;i<ITEMS_PER_WI;i++) {
			if(c+i < over) {
				val = max(val,inp[c + i + inp_offset_]);
			}
		}
		
		my_work_group_reduce_max(val, WG_SIZE);
	}
	else
	{
		val = 0;
		for(uint i=0;i<ITEMS_PER_WI;i++) {
			if(c+i < over) {
				val += inp[c+i + inp_offset_];
			}
		}
		
		my_work_group_reduce_add(val, WG_SIZE);
		val = val * scale;
	}

	if(get_local_id(1) == 0)
		outp[outp_offset_] = val;
}
