#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer mean_buf { float mean[]; };
	layout(binding = 1, std430) readonly buffer var_buf { float var[]; };
	layout(binding = 2, std430) readonly buffer gamma_buf { float gamma[]; };
	layout(binding = 3, std430) readonly buffer beta_buf { float beta[]; };
	layout(binding = 4, std430) writeonly buffer a_buf { float a[]; };
	layout(binding = 5, std430) writeonly buffer b_buf { float b[]; };
#endif

layout(push_constant, std430) uniform combine_mean_var_with_gamma_beta
{
	uint N;float eps;
#if USE_BDA
	__global float const * mean;
#endif
	uint  mean_offset;
#if USE_BDA
	__global float const * var;
#endif
	uint  var_offset;
#if USE_BDA
	__global float const * gamma;
#endif
	uint  gamma_offset;
#if USE_BDA
	__global float const * beta;
#endif
	uint  beta_offset;
#if USE_BDA
	__global float *a;
#endif
	uint  a_offset;
#if USE_BDA
	__global float *b;
#endif
	uint  b_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= N)
        return;
    float scale = 1.0f / sqrt(var[pos + var_offset] + eps);
    float offset  = - mean[pos + mean_offset] * scale;
    float G = gamma[pos + gamma_offset];
    scale *= G;
    offset = offset * G + beta[pos + beta_offset];
    a[pos + a_offset] = scale;
    b[pos + b_offset] = offset;
}
