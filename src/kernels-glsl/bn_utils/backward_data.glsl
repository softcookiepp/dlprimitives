#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x_buf { dtype x[]; };
	layout(binding = 1, std430) readonly buffer dy_buf { dtype dy[]; };
	layout(binding = 2, std430) readonly buffer fx_buf { dtype fx[]; };
	layout(binding = 3, std430) readonly buffer fdy_buf { dtype fdy[]; };
	layout(binding = 4, std430) readonly buffer b_buf { dtype b[]; };
	layout(binding = 5, std430) buffer dx_buf { dtype dx[]; };
#endif

layout(push_constant) uniform backward_data
{
	uint batches; uint channels; uint HW;
#if USE_BDA
	__global dtype const *x;
#endif
	uint  x_offset;
#if USE_BDA
	__global dtype const *dy;
#endif
	uint  dy_offset;
#if USE_BDA
	__global dtype const *fx;
#endif
	uint  fx_offset;
#if USE_BDA
	__global dtype const *fdy;
#endif
	uint  fdy_offset;
#if USE_BDA
	__global dtype const *b;
#endif
	uint  b_offset;
#if USE_BDA
	__global dtype *dx;
#endif
	uint  dx_offset;
	dtype factor;
};

void main()
{
    uint batch  = get_global_id(DIM_B);
    uint f  = get_global_id(DIM_F);
    uint rc = get_global_id(DIM_RC);
    if(batch >= batches || f >= channels || rc >= HW)
        return;
    uint pos = (batch * channels + f) * HW + rc;
    dtype grad =  fx[fx_offset + f] * x[x_offset + pos]  + fdy[fdy_offset + f] * dy[dy_offset + pos] + b[b_offset + f];
    if(factor == 0)
        dx[dx_offset + pos] = grad;
    else
        dx[dx_offset + pos] = dx[dx_offset + pos] * factor + grad;
}
