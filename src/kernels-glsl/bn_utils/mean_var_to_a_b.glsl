#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer mean_buf { dtype mean[]; };
	layout(binding = 1, std430) readonly buffer var_buf { dtype var[]; };
	layout(binding = 2, std430) buffer a_buf { dtype a[]; };
	layout(binding = 3, std430) buffer b_buf { dtype b[]; };
#endif

layout(push_constant, std430) uniform mean_var_to_a_b
{
	int N;dtype eps;
#if USE_BDA
	__global dtype const * mean;
#endif
	uint mean_offset;
#if USE_BDA
	__global dtype const * var;
#endif
	uint var_offset;
#if USE_BDA
	__global dtype *a;
#endif
	uint a_offset;
#if USE_BDA
	__global dtype *b;
#endif
	uint b_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= N)
        return;
    dtype scale = 1.0f / sqrt(var[pos + var_offset] + eps);
    dtype offset  = - mean[pos + mean_offset] * scale;
    a[pos + a_offset] = scale;
    b[pos + b_offset] = offset;
}
