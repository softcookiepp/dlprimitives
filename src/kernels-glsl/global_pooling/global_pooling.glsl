#version 450
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

layout(local_size_x = 1, local_size_y = WG_SIZE, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { dtype inp[]; };
	layout(binding = 0, std430) writeonly buffer outp_buf { dtype outp[]; };
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
    
    //REDUCE_PREPARE(WG_SIZE,dtype);

#if POOL_MODE == 0

    dtype val = -DTYPE_MAX;
    for(uint i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val = max(val,inp[c + i + inp_offset_]);
        }
    }
    
    my_work_group_reduce_max(val, WG_SIZE);
#else
    dtype val = 0;
    for(uint i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < over) {
            val += inp[c+i + inp_offset_];
        }
    }
    
    my_work_group_reduce_add(val);
    val = val * scale;
#endif

    if(get_local_id(1) == 0)
        outp[outp_offset_] = val;
}
