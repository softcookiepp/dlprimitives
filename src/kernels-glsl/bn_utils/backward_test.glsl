#version 450

#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer dx_buf { dtype dx[]; };
	layout(binding = 1, std430) readonly buffer dy_buf { dtype dy[]; };
	layout(binding = 2, std430) readonly buffer a_buf { dtype a[]; };
#endif

layout(push_constant, std430) uniform backward_test
{
	uint batches; uint channels; uint HW;
#if USE_BDA
	dtype_addr_rw dx;
#endif
	uint dx_offset;
#if USE_BDA
	dtype_addr_ro dy;
#endif
	uint dy_offset;
#if USE_BDA
	dtype_addr_ro a;
#endif
	uint a_offset;
	dtype factor;
};
        
void main()     
{
    uint b  = get_global_id(DIM_B);
    uint f  = get_global_id(DIM_F);
    uint rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    uint pos = (b * channels + f) * HW + rc;
    dtype val = dy[pos + dy_offset] * a[f + a_offset];
    if(factor == 0)
        dx[pos + dx_offset] = val;
    else
        dx[pos + dx_offset] = dx[pos + dx_offset]*factor + val;
}
