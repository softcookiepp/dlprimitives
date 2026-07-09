#version 450
#include "softmax_common.glsl"


__kernel 
__attribute__((reqd_work_group_size(1,WG_SIZE,1)))
void softmax_backward(int batch,int channels,int extra_batch,
             __global dtype *in,ulong  data_offset,
             __global dtype const *out,ulong  out_offset,
             __global dtype const *out_diff,ulong  out_diff_offset,
             dtype factor
             )
{
    in += data_offset;
    out += out_offset;
    out_diff += out_diff_offset;
    
    long b = get_global_id(0);
    long eb = get_global_id(2);

    if(b >= batch)
        return;
    if(eb >= extra_batch)
        return;
    long step = extra_batch;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * channels * extra_batch + eb;
    out += b * channels * extra_batch + eb;
    out_diff += b * channels * extra_batch + eb;
    
    dtype sum = 0;
    REDUCE_PREPARE(WG_SIZE,dtype);

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
    #pragma unroll(ITEMS_PER_WI)
    #else
    #pragma unroll(LOCAL_ITEMS_LIMIT)
    #endif
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c+i < channels) {
            #if LOG_SM == 1
            sum += out_diff[(c+i) * step];
            #else
            sum += out_diff[(c+i)*step] * out[(c+i)*step];
            #endif
        }
    }

    my_work_group_reduce_add(sum);

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
    #pragma unroll(ITEMS_PER_WI)
    #else
    #pragma unroll(LOCAL_ITEMS_LIMIT)
    #endif
    for(int i=0;i<ITEMS_PER_WI;i++) {
        if(c + i < channels) {
            #if LOG_SM == 1
            float dxval = out_diff[(c+i)*step] - exp(out[(c+i)*step]) * sum;
            #else
            float dxval = (out_diff[(c+i)*step] - sum) * out[(c+i)*step]; 
            #endif
            if(factor == 0)
                in[(c+i)*step] = dxval;
            else
                in[(c+i)*step] = factor * in[(c+i)*step] + dxval;
        }
    }
}
