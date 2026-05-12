#version 450
#include "softmax-common.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { dtype inp[]; };
	layout(binding = 1, std430) buffer outp_buf { dtype outp[]; };
#endif

layout(push_constant, std430) uniform softmax
{
	uint batch;
	uint channels;
	uint extra_batch;
#if USE_BDA
	__global dtype const *in,
#endif
	uint  data_offset;
#if USE_BDA
	__global dtype *out,
#endif
	uint  out_offset;
};

REDUCE_PREPARE(WG_SIZE, dtype);

void main()
{
    uint inp_ = data_offset;
    uint outp_ = out_offset;
    
    uint b = get_global_id(0);
    uint eb = get_global_id(2);
    uint step = extra_batch;

    if(b >= batch)
        return;
    if(eb >= extra_batch)
        return;

    uint c = get_global_id(1) * ITEMS_PER_WI;

    inp_ += (b * channels * extra_batch + eb);
    outp_ += (b * channels * extra_batch + eb);
    
    // REDUCE_PREPARE(WG_SIZE, dtype);
    REDUCE_FILL(DTYPE_MIN);

    dtype val = DTYPE_MIN;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        dtype values[ITEMS_PER_WI];
        UNROLL(ITEMS_PER_WI)
        for(uint i=0; i < ITEMS_PER_WI; i++)
        {
            if(c+i < channels)
            {
				dtype vi = inp[(c+i)*step + inp_];
				val = max(val, vi);
                values[i] = inp[(c+i)*step + inp_];
                val = max(val, values[i]);
            }
        }
    #else
        for(uint i=0;i<ITEMS_PER_WI;i++)
        {
            if(c+i < channels)
            {
                val = max(val, inp[(c+i)*step + inp_]);
            }
        }
    #endif


    my_work_group_reduce_max(val, WG_SIZE);
    dtype maxv = val;

    dtype sum = 0;

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        UNROLL(ITEMS_PER_WI)
        for(uint i=0; i < ITEMS_PER_WI; i++) {
            if(c+i < channels) {
                #if LOG_SM == 1
					sum += exp(values[i] - maxv);
                #else
					values[i] = exp(values[i] - maxv);
					sum += values[i];
                #endif
            }
        }
    #else
		UNROLL(ITEMS_PER_WI)
        for(uint i=0; i < ITEMS_PER_WI; i++)
        {
            if(c+i < channels)
            {
                dtype tmp = exp(inp[(c+i)*step + inp_] - maxv);
                #if LOG_SM == 0
					outp[(c+i)*step + outp_] = tmp;
                #endif
                sum += tmp;
            }
        }
    #endif
    my_work_group_reduce_add(sum, WG_SIZE);
    barrier();

    #if LOG_SM == 0
		val = dtype(1) / sum;
    #else
		val = (-1.0)*log(sum);
    #endif

    #if ITEMS_PER_WI <= LOCAL_ITEMS_LIMIT
        UNROLL(ITEMS_PER_WI)
        for(uint i = 0; i < ITEMS_PER_WI; i++)
        {
            if(c + i < channels)
            {
                #if LOG_SM == 1
					outp[(c+i)*step + outp_] = values[i] - maxv + val;
                #else
					outp[(c+i)*step + outp_] = values[i] * val;
                #endif
            }
        }
    #else
		UNROLL(ITEMS_PER_WI)
        for(uint i=0; i < ITEMS_PER_WI; i++)
        {
            if(c + i < channels)
            {
                #if LOG_SM == 1
					outp[(c+i)*step + outp_] = inp[(c+i)*step + inp_] - maxv + val;
                #else
					outp[(c+i)*step + outp_] *= val;
                #endif
            }
        }
    #endif
}
