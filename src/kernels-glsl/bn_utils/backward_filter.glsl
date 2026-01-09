#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer mean_buf { float mean[]; };
	layout(binding = 1, std430) readonly buffer var_buf { float var[]; };
	layout(binding = 2, std430) readonly buffer dy_sum_buf { float dy_sum[]; };
	layout(binding = 3, std430) readonly buffer dyx_sum_buf { float dyx_sum[]; };
	layout(binding = 4, std430) buffer dgamma_buf { float dgamma[]; };
	layout(binding = 5, std430) buffer dbeta_buf { float dbeta[]; };
#endif

layout(push_constant, std430) uniform backward_filter
{
	int N;
#if USE_BDA
	__global float const *mean;
#endif
	uint  mean_offset;
#if USE_BDA
	__global float const *var;
#endif
	uint  var_offset;
#if USE_BDA
	__global float const *dy_sum;
#endif
	uint  dy_sum_offset;
#if USE_BDA
	__global float const *dyx_sum;
#endif
	uint  dyx_sum_offset;
#if USE_BDA
	__global float *dgamma;
#endif
	uint dgamma_offset;
	bool use_gamma; // here we are again
#if USE_BDA
	__global float *dbeta;
#endif
	uint dbeta_offset;
	bool use_beta;
	float eps;
	float factor_gamma;
	float factor_beta;
};

void main()
{
    uint i=get_global_id(0);
    if(i >= N)
        return;

    float dys = dy_sum[i + dy_sum_offset];

    if(use_gamma)
    {
        float dG = (dyx_sum[i + dyx_sum_offset] - mean[i + mean_offset]*dys) / sqrt(var[i + var_offset] + eps); 
        if(factor_gamma == 0)
            dgamma[i + dgamma_offset] = dG;
        else
            dgamma[i + dgamma_offset] = dgamma[i + dgamma_offset]*factor_gamma + dG;
    }
    
    if(use_beta)
    {
        if(factor_beta == 0)
            dbeta[i + dbeta_offset] = dys;
        else
            dbeta[i + dbeta_offset] = dbeta[i + dbeta_offset] * factor_beta + dys;
    }
}
