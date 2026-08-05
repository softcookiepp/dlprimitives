#version 450
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer data_buf { dtype data[]; };
	layout(binding = 1, std430) readonly buffer label_buf { itype label[]; };
	layout(binding = 2, std430) writeonly buffer outp_buf { dtype outp[]; };
#endif


layout(push_constant, std430) uniform nll_loss_forward
{
	uint batch;
	uint channels;
#if USE_BDA
	dtype_addr_ro data;
#endif
	uint  data_offset;
#if USE_BDA
	itype_addr_ro label;
#endif
	uint  label_offset;
#if USE_BDA
	dtype_addr_rw outp;
#endif
	uint  outp_offset;
	dtype scale;
};

//#if REDUCE == 1
    REDUCE_PREPARE(WG_SIZE, dtype);
//#endif

void main()
{
    uint data_ = data_offset;
    uint label_ = label_offset;
    uint outp_ = outp_offset;
    
    uint item = get_local_id(0) * ITEMS_PER_WI;
    #if REDUCE == 1
    dtype sum = 0;
    #endif

    UNROLL(ITEMS_PER_WI)
    for(uint i=0;i<ITEMS_PER_WI;i++,item++) {
        if(item < batch) {
            uint index = uint(label[item + label_]);
            dtype loss_value = 0;
            if(0<= index && index < channels) {
                loss_value = -data[item*channels + index + data_];
            }
            #if REDUCE==0
            outp[item + outp_] = loss_value * scale;
            #else
            sum += loss_value;
            #endif
        }
    }
#if REDUCE == 1
    my_work_group_reduce_add(sum, WG_SIZE);
    if(get_local_id(0) == 0) {
        outp[outp_] = sum * scale;
    }
#endif
}


