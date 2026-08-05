#version 450
#include "../common/defs.glsl"
#include "bn_defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer cur_mean_buf { dtype cur_mean[]; };
	layout(binding = 1, std430) readonly buffer cur_var_buf { dtype cur_var[]; };
	layout(binding = 2, std430) buffer run_mean_buf { dtype run_mean[]; };
	layout(binding = 3, std430) buffer run_var_buf { dtype run_var[]; };
#endif

layout(push_constant, std430) uniform update_sums
{
	uint N;
#if USE_BDA
	__global dtype const * restrict cur_mean;
#endif
	uint  cur_mean_offset;
#if USE_BDA
	__global dtype const * restrict cur_var ;
#endif
	uint  cur_var_offset;
#if USE_BDA
	__global dtype * restrict run_mean;
#endif
	uint  run_mean_offset;
#if USE_BDA
	__global dtype * restrict run_var ;
#endif
	uint  run_var_offset;
	dtype cur_mean_factor;dtype run_mean_factor;
	dtype cur_var_factor; dtype run_var_factor;
};

void main()
{
    uint p = get_global_id(0);
    if(p >= N)
        return;

    run_mean[p + run_mean_offset] = cur_mean[p + cur_mean_offset] * cur_mean_factor + run_mean[p + run_mean_offset] * run_mean_factor;
    run_var[p + run_var_offset]  = cur_var[p + cur_var_offset]  * cur_var_factor  + run_var[p + run_var_offset]  * run_var_factor;
}
