#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer tensor_buf { dtype tensor[]; };
	layout(binding = 1, std430) readonly buffer bias_buf { dtype bias[]; };
#endif


layout(push_constant, std430) uniform fwd_bias
{
	uint B;uint F;uint RC;
#if USE_BDA
	dtype_addr_rw tensor;
#endif
	uint tensor_offset;
#if USE_BDA
	dtype_addr_ro bias;
#endif
	uint bias_offset;
};

void main()
{
    uint rc = get_global_id(0);
    uint f  = get_global_id(1);
    uint b  = get_global_id(2);
    if(rc >= RC || f >= F || b >= B)
        return;
    tensor[b * (F*RC) + f * RC + rc + tensor_offset] += bias[f + bias_offset];
}
