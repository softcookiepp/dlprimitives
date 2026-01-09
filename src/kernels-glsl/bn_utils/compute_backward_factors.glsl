#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer mean_buf { float mean[]; };
	layout(binding = 1, std430) readonly buffer varrstd_buf { float varrstd[]; };
	layout(binding = 2, std430) readonly buffer dy_sum_buf { float dy_sum[]; };
	layout(binding = 3, std430) readonly buffer dyx_sum_buf { float dyx_sum[]; };
	layout(binding = 4, std430) buffer gamma_in_buf { float gamma_in[]; };
	layout(binding = 5, std430) buffer x_factor_buf { float x_factor[]; };
	layout(binding = 5, std430) buffer dy_factor_buf { float dy_factor[]; };
	layout(binding = 5, std430) buffer offset_buf { float offset[]; };
#endif


layout(push_constant, std430) uniform compute_backward_factors
{
	int N;int M;float eps;
#if USE_BDA
	__global float const *mean;
#endif
	uint  mean_offset;
#if USE_BDA
	__global float const *varrstd;
#endif
	uint  varrstd_offset;
#if USE_BDA
	__global float const *dy_sum;
#endif
	uint  dy_sum_offset;
#if USE_BDA
	__global float const *dyx_sum;
#endif
	uint  dyx_sum_offset;
#if USE_BDA
	__global float const *gamma_in;
#endif
	uint  gamma_in_offset;
	bool use_gamma;
#if USE_BDA
	__global float *x_factor;
#endif
	uint  x_factor_offset;
#if USE_BDA
	__global float *dy_factor;
#endif
	uint  dy_factor_offset;
#if USE_BDA
	__global float *offset;
#endif
	uint  offset_offset;
};

void main()
{
    uint i = get_global_id(0);
    if(i >= N)
        return;

    float one_by_M = 1.0f / M;
    float rsqrtsig;
    if(eps < 0)
        rsqrtsig = varrstd[i + varrstd_offset];
    else
        rsqrtsig = 1.0f / sqrt(varrstd[i + varrstd_offset] + eps);

    float gamma=1.0f;
    if(use_gamma)
        gamma = gamma_in[i + gamma_in_offset];
    float mu = mean[i + mean_offset];
    float dys = dy_sum[i + dy_sum_offset];
    float dsig = -0.5 * gamma * (dyx_sum[i + dyx_sum_offset] - mu * dys) * (rsqrtsig * rsqrtsig * rsqrtsig);
    float gamma_div_sigsqrt = gamma * rsqrtsig;
    float dmu = -dys * gamma_div_sigsqrt;
    float F_dy = gamma_div_sigsqrt;
    float F_x  = 2*dsig * one_by_M;
    float B = one_by_M * (dmu - dsig * 2 * mu);

    dy_factor[i + dy_factor_offset] = F_dy;
    x_factor[i + x_factor_offset] = F_x;
    offset[i + offset_offset] = B;
}
