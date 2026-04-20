#version 450
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#ifndef LOG_SM
#define LOG_SM 0
#endif

#ifndef CALC_LOSS
#define CALC_LOSS 0
#endif

#if CALC_LOSS==1
#include "../common/atomic.glsl"
#endif

layout(local_size_x = 1, local_size_y = WG_SIZE, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer inp_buf {dtype inp[]; };
	layout(binding = 1, std430) readonly buffer outp_buf {dtype outp[]; };
	layout(binding = 2, std430) readonly buffer out_diff_buf {dtype out_diff[]; };
#endif

layout() uniform softmax_backward
{
	uint batch;
	uint channels;
	uint extra_batch;
#if USE_BDA
	__global dtype *in;
#endif
	uint  data_offset;
#if USE_BDA
	__global dtype const *out;
#endif
	uint  out_offset;
#if USE_BDA
	__global dtype const *out_diff;
#endif
	uint out_diff_offset;
	dtype factor;
};

REDUCE_PREPARE(WG_SIZE,dtype);

void main()
{
    uint inp_ = data_offset;
    uint outp_ = out_offset;
    uint out_diff_ += out_diff_offset;
    
    uint b = get_global_id(0);
    uint eb = get_global_id(2);

    if(b >= batch)
        return;
    if(eb >= extra_batch)
        return;
    uint step = extra_batch;

    uint c = get_global_id(1) * ITEMS_PER_WI;

    inp_ += b * channels * extra_batch + eb;
    outp_ += b * channels * extra_batch + eb;
    out_diff_ += b * channels * extra_batch + eb;
    
    dtype sum = 0;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
		UNROLL(ITEMS_PER_WI)
    #else
		UNROLL(LOCAL_ITEMS_LIMIT)
    #endif
    for(uint i = 0; i < ITEMS_PER_WI; i++)
    {
        if(c+i < channels)
        {
            #if LOG_SM == 1
				sum += out_diff[(c+i) * step + out_diff_];
            #else
				sum += out_diff[(c+i)*step + out_diff_] * outp[(c+i)*step + outp_];
            #endif
        }
    }

    my_work_group_reduce_add(sum);

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
		UNROLL(ITEMS_PER_WI)
    #else
		UNROLL(LOCAL_ITEMS_LIMIT)
    #endif
    for(uint i = 0; i < ITEMS_PER_WI; i++)
    {
        if(c + i < channels)
        {
            #if LOG_SM == 1
				float dxval = out_diff[(c+i)*step + out_diff_] - exp(outp[(c+i)*step + outp_]) * sum;
            #else
				float dxval = (out_diff[(c+i)*step + out_diff_] - sum) * outp[(c+i)*step + outp_]; 
            #endif
            if(factor == 0)
                inp[(c+i)*step + inp_] = dxval;
            else
                inp[(c+i)*step + inp_] = factor * inp[(c+i)*step + inp_] + dxval;
        }
    }
}
