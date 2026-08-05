#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer mean_buf { dtype mean[]; };
	layout(binding = 1, std430) readonly buffer var_buf { dtype var[]; };
	layout(binding = 2, std430) readonly buffer gamma_buf { dtype gamma[]; };
	layout(binding = 3, std430) readonly buffer beta_buf { dtype beta[]; };
	layout(binding = 4, std430) writeonly buffer a_buf { dtype a[]; };
	layout(binding = 5, std430) writeonly buffer b_buf { dtype b[]; };
#endif

layout(push_constant, std430) uniform combine_mean_var_with_gamma_beta
{
	uint N;dtype eps;
#if USE_BDA
	__global dtype const * mean;
#endif
	uint  mean_offset;
#if USE_BDA
	__global dtype const * var;
#endif
	uint  var_offset;
#if USE_BDA
	__global dtype const * gamma;
#endif
	uint  gamma_offset;
#if USE_BDA
	__global dtype const * beta;
#endif
	uint  beta_offset;
#if USE_BDA
	__global dtype *a;
#endif
	uint  a_offset;
#if USE_BDA
	__global dtype *b;
#endif
	uint  b_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= N)
        return;
    dtype scale = 1.0f / sqrt(var[pos + var_offset] + eps);
    dtype offset  = - mean[pos + mean_offset] * scale;
    dtype G = gamma[pos + gamma_offset];
    scale *= G;
    offset = offset * G + beta[pos + beta_offset];
    a[pos + a_offset] = scale;
    b[pos + b_offset] = offset;
}
