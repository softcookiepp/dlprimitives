#version 450

__kernel
void update_sums(int N,
                 __global float const * restrict cur_mean,ulong  cur_mean_offset,
                 __global float const * restrict cur_var ,ulong  cur_var_offset,
                 __global float * restrict run_mean,ulong  run_mean_offset,
                 __global float * restrict run_var ,ulong  run_var_offset,
                 float cur_mean_factor,float run_mean_factor,
                 float cur_var_factor, float run_var_factor)
{
    int p = get_global_id(0);
    if(p >= N)
        return;
    run_mean += run_mean_offset;
    run_var  += run_var_offset;
    cur_mean += cur_mean_offset;
    cur_var  += cur_var_offset;

    run_mean[p] = cur_mean[p] * cur_mean_factor + run_mean[p] * run_mean_factor;
    run_var[p]  = cur_var[p]  * cur_var_factor  + run_var[p]  * run_var_factor;
}
