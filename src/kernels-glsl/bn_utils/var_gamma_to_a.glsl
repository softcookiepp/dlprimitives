#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer var_buf { float var[]; };
	layout(binding = 1, std430) readonly buffer gamma_buf { float gamma[]; };
	layout(binding = 2, std430) buffer a_buf { float a[]; };
#endif

layout(push_constant, std430) uniform var_gamma_to_a
{
	int N;float eps;
#if USE_BDA
	__global float const * var;
#endif
	uint var_offset;
#if USE_BDA
	__global float const * gamma;
#endif
	uint gamma_offset;
	bool use_gamma; // added since buffers can't be evaluated as boolean statements
#if USE_BDA
	__global float *a;
#endif
	uint a_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= N)
        return;
    float scale = 1.0f / sqrt(var[pos + var_offset] + eps);
    if(use_gamma)
        scale *= gamma[pos + gamma_offset];
    a[pos + a_offset] = scale;
}












