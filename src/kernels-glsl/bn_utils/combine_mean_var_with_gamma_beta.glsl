#version 450
__kernel
void combine_mean_var_with_gamma_beta(
                     int N,float eps,
                     __global float const * mean,ulong  mean_offset,
                     __global float const * var, ulong  var_offset,
                     __global float const * gamma,ulong  gamma_offset,
                     __global float const * beta,ulong  beta_offset,
                     __global float *a,ulong  a_offset,
                     __global float *b,ulong  b_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    gamma += gamma_offset;
    beta += beta_offset;
    a += a_offset;
    b += b_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    float offset  = - mean[pos] * scale;
    float G = gamma[pos];
    scale *= G;
    offset = offset * G + beta[pos];
    a[pos] = scale;
    b[pos] = offset;
}
