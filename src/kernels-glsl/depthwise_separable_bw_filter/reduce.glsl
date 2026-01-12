#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#include "defs.glsl"

//#if SECOND_REDUCE_SIZE > 1
layout(local_size_x = SECOND_REDUCE_SIZE, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer partial_values_buf { float partial_values[]; };
	layout(binding = 1, std430) buffer sums_buf { float sums[]; };
#endif

layout(push_constant, std430) uniform reduce
{
#if USE_BDA
	__global float const * restrict partial_values;
#endif
	uint partial_values_offset;
#if USE_BDA
	__global float * restrict sums;
#endif
	uint sums_offset;
	float factor;
};

REDUCE_PREPARE(SECOND_REDUCE_SIZE,float);

void main()
{
    //sums += sums_offset;
    //partial_values += partial_values_offset;
    uint k = get_global_id(1);
    if(k > KERN * KERN * CHANNELS)
        return;
    
    float val = partial_values[k + get_local_id(0) * STRIDE + partial_values_offset];

    my_work_group_reduce_add(val);

    if(get_local_id(0) == 0) {
        if(factor == 0)
            sums[k + sums_offset] = val;
        else
            sums[k + sums_offset] = fma(sums[k + sums_offset],factor,val);
    }    
}

//#endif
