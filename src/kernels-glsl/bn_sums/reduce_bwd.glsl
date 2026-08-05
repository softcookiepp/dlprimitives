#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#define USE_SPEC_CONSTANTS 1
#include "../common/reduce2.glsl"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef BACKWARD
#define BACKWARD 1
#endif

// this might get confusing.
// #if SECOND_REDUCE_SIZE > 1
layout(local_size_x = SECOND_REDUCE_SIZE, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer s1_buf { float s1[]; };
	layout(binding = 1, std430) readonly buffer s2_buf { float s2[]; };
	#if BACKWARD == 1
		layout(binding = 2, std430) buffer dyx_sum_buf { float dyx_sum[]; };
		layout(binding = 3, std430) buffer dy_sum_buf { float dy_sum[]; };
	#else
		layout(binding = 2, std430) buffer x_mean_buf { float x_mean[]; };
		layout(binding = 3, std430) buffer x_var_buf { float x_var[]; };
	#endif
#endif

REDUCE_PREPARE_X2(SECOND_REDUCE_SIZE,float);

layout(push_constant, std430) uniform reduce
{
	uint channels;
	#if USE_BDA
		__global float const * restrict s1;
	#endif
	uint s1_offset;
	#if USE_BDA
		__global float const * restrict s2;
	#endif
	uint s2_offset;
#if BACKWARD == 1
	#if USE_BDA
		__global float *dyx_sum;
	#endif
	uint dyx_sum_offset;
	#if USE_BDA
		__global float *dy_sum;
	#endif
	uint dy_sum_offset;
#else
	#if USE_BDA
		__global float *x_mean;
	#endif
	uint x_mean_offset;
	#if USE_BDA
		__global float *x_var;
	#endif
	uint x_var_offset;
	float one_div_M;
#endif
};

void main()
{
    uint f = get_global_id(1);
    if(f >= channels)
        return;
    
    // REDUCE_PREPARE_X2(SECOND_REDUCE_SIZE,float);

    uint read_pos = f + get_local_id(0) * channels;
    vec2 sum;
    sum[0] = s1[read_pos + s1_offset];
    sum[1] = s2[read_pos + s2_offset];
    
    my_work_group_reduce_add_x2(sum, SECOND_REDUCE_SIZE);

    if(get_local_id(0) == 0)
    {
        #if BACKWARD == 0
			float mean_val  = sum[0] * one_div_M;
			float mean2_val = sum[1] * one_div_M;
			x_mean[f + x_mean_offset] = mean_val;
			 x_var[f + x_var_offset] = mean2_val - mean_val*mean_val;
        #else
			dyx_sum[f +  dyx_sum_offset] = sum[0];
			 dy_sum[f + dy_sum_offset] = sum[1];
        #endif
    }    
}

//#endif
