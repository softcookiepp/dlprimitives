#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/activation.glsl"
PREPARE_ACTIVATION(3)

#if USE_BDA == 0
	layout(binding = 0, std430) buffer y_buf { dtype y[]; };
	layout(binding = 1, std430) buffer dy_buf { dtype dy[]; };
	layout(binding = 2, std430) buffer dx_buf { dtype dx[]; };
#endif

layout(push_constant, std430) uniform activation_diff
{
	uint size;
#if USE_BDA
	dtype_addr_rw y;
#endif
	uint y_offset;
#if USE_BDA
	dtype_addr_rw dy;
#endif
	uint dy_offset;
#if USE_BDA
	dtype_addr_rw dx;
#endif
	uint dx_offset;
	dtype beta;
};
	
void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
    dtype y_val  = y[pos + y_offset];
    dtype dy_val = dy[pos + dy_offset];
    dtype diff = ACTIVATION_FINV(y_val, dy_val);
    if(beta == 0)
        dx[pos + dx_offset] = diff;
    else
        dx[pos + dx_offset] = fma(dx[pos + dx_offset], beta, diff);
}


