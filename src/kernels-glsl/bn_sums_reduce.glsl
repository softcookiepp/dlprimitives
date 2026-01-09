///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.h"
#include "reduce2.h"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef BACKWARD
#define BACKWARD 0
#endif

// this might get confusing.
#if SECOND_REDUCE_SIZE > 1
__kernel
__attribute__((reqd_work_group_size(SECOND_REDUCE_SIZE,1,1)))
void reduce(int channels,
            __global float const * restrict s1,ulong s1_offset,
            __global float const * restrict s2,ulong s2_offset,
#if BACKWARD == 1
            __global float *dyx_sum,ulong dyx_sum_offset,
            __global float *dy_sum,ulong dy_sum_offset
#else
            __global float *x_mean,ulong x_mean_offset,
            __global float *x_var, ulong x_var_offset,
            float one_div_M
#endif
      )      
{
    s1 += s1_offset;
    s2 += s2_offset;
#if BACKWARD == 1
    dyx_sum += dyx_sum_offset;
    dy_sum  += dy_sum_offset;
#else
    x_mean += x_mean_offset;
    x_var  += x_var_offset;
#endif

    int f = get_global_id(1);
    if(f >= channels)
        return;
    
    REDUCE_PREPARE_X2(SECOND_REDUCE_SIZE,float);

    int read_pos = f + get_local_id(0) * channels;
    float2 sum;
    sum.s0 = s1[read_pos];
    sum.s1 = s2[read_pos];
    
    my_work_group_reduce_add_x2(sum);

    if(get_local_id(0) == 0) {
        #if BACKWARD == 0
        float mean_val  = sum.s0 * one_div_M;
        float mean2_val = sum.s1 * one_div_M;
        x_mean[f] = mean_val;
         x_var[f] = mean2_val - mean_val*mean_val;
        #else
        dyx_sum[f] = sum.s0;
         dy_sum[f] = sum.s1;
        #endif
    }    
}

#endif
