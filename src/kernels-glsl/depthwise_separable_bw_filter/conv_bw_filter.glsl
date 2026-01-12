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

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { float inp[]; };
	layout(binding = 1, std430) buffer kern_buf { float kern[]; };
	layout(binding = 1, std430) readonly buffer outp_buf { float outp[]; };
#endif

layout(push_constant, std430) uniform conv_bw_filter
{
	uint batch;uint height;uint width;
#if USE_BDA
	__global float const *inp;
#endif
	uint inp_offset;
#if USE_BDA
	__global float *kern;
#endif
	uint kernel_offset;
#if USE_BDA
	__global float const *outp;
#endif
	uint outp_offset;
#if SECOND_REDUCE_SIZE == 1
	float factor;
#endif          
};

REDUCE_PREPARE(WG_SIZE,float);

void main()
{
    uint _inp_offset = inp_offset;
    uint _outp_offset = outp_offset;
    uint _kernel_offset = kernel_offset;

    uint k = get_global_id(1);
    if( k > KERN * KERN * CHANNELS)
        return;

    uint dk = k % (KERN * KERN);
    uint d  = k / (KERN * KERN); 

    uint dr = dk / KERN;
    uint dc = dk % KERN;

    _inp_offset  += d * (width * height);
    _outp_offset += d * (width * height);

    uint items = batch * (width * height);
    const uint wg_size2 = WG_SIZE * SECOND_REDUCE_SIZE;
    uint items_per_wg = (items + wg_size2 - 1) / wg_size2;
    uint my_start = items_per_wg * get_global_id(0); // it is same as local id for 1stage reduce
    uint my_end   = min(my_start + items_per_wg,items);

    float sum = 0;
    uint b  = my_start / (width * height);
    uint rc = my_start % (width * height);
    uint r = rc / width;
    uint c = rc % width;

    //#pragma unroll(16)
    for(uint index = my_start;index <my_end;index ++) {
        uint sr = r - KERN/2 + dr;
        uint sc = c - KERN/2 + dc;
        if(b < batch && 0<=sr && sr < height && 0 <= sc && sc < width) {
            float y = outp[b*(CHANNELS * height * width) + r  * width +c + _outp_offset];
            float x =  inp[b*(CHANNELS * height * width) + sr * width +sc + _inp_offset];
            sum += x*y;
        }
        c++;
        if(c == width) {
            c = 0;
            r++;
            if(r == height) {
                r = 0;
                b ++;
            }
        }
    }

    my_work_group_reduce_add(sum, WG_SIZE);

    if(get_local_id(0) == 0) {
        #if SECOND_REDUCE_SIZE == 1
        if(factor == 0)
            kern[k + _kernel_offset] = sum;
        else
            kern[k + _kernel_offset] = fma(kern[k + _kernel_offset],factor,sum);
        #else
            #define STRIDE (KERN * KERN * CHANNELS)
            kern[k + STRIDE * get_group_id(0) + _kernel_offset] = sum;
        #endif
    }

}
