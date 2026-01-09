#version 450

__kernel
void var_gamma_to_a(int N,float eps,
              __global float const * var, ulong  var_offset,
              __global float const * gamma, ulong  gamma_offset,
              __global float *a,ulong  a_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    var  += var_offset;
    gamma += gamma_offset;
    a += a_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    if(gamma)
        scale *= gamma[pos];
    a[pos] = scale;
}












