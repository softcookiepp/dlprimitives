#version 450
#include "softmax-common.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { dtype inp[]; };
	layout(binding = 1, std430) writeonly buffer outp_buf { dtype outp[]; };
#endif

layout(push_constant, std430) uniform softmax
{
	int batch,
	int channels,
	int extra_batch,
#if USE_BDA
	__global dtype const *in,
#endif
	ulong  data_offset,
#if USE_BDA
	__global dtype *out,
#endif
	ulong  out_offset
};

void main()
{
    inp += data_offset;
    outp += out_offset;
    
    long b = get_global_id(0);
    long eb = get_global_id(2);
    long step = extra_batch;

    if(b >= batch)
        return;
    if(eb >= extra_batch)
        return;

    int c = get_global_id(1) * ITEMS_PER_WI;

    in += b * channels * extra_batch + eb;
    out += b * channels * extra_batch + eb;
    
    REDUCE_PREPARE(WG_SIZE,dtype);

    dtype val = -DTYPE_MAX;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        dtype values[ITEMS_PER_WI];
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                values[i] = in[(c+i)*step];
                val = max(val,values[i]);
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                val = max(val,in[(c+i)*step]);
            }
        }
    #endif


    my_work_group_reduce_max(val);
    dtype maxv = val;

    dtype sum = 0;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                #if LOG_SM == 1
                sum += exp(values[i] - maxv);
                #else
                sum += values[i] = exp(values[i] - maxv);
                #endif
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c+i < channels) {
                dtype tmp = exp(in[(c+i)*step] - maxv);
                #if LOG_SM == 0
                out[(c+i)*step] = tmp;
                #endif
                sum += tmp;
            }
        }
    #endif
    my_work_group_reduce_add(sum);

    #if LOG_SM == 0
    val = (dtype)1 / sum;
    #else
    val = -log(sum);
    #endif

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        #pragma unroll
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                #if LOG_SM == 1
                out[(c+i)*step] = values[i] - maxv + val;
                #else
                out[(c+i)*step] = values[i] * val;
                #endif
            }
        }
    #else
        for(int i=0;i<ITEMS_PER_WI;i++) {
            if(c + i < channels) {
                #if LOG_SM == 1
                out[(c+i)*step] = in[(c+i)*step] - maxv + val;
                #else
                out[(c+i)*step] *= val;
                #endif
            }
        }
    #endif
}
