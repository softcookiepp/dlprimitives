__kernel
void compute_backward_factors(int N,int M,float eps,
                              __global float const *mean,ulong  mean_offset,
                              __global float const *varrstd, ulong  varrstd_offset,
                              __global float const *dy_sum, ulong  dy_sum_offset,
                              __global float const *dyx_sum,ulong  dyx_sum_offset,
                              __global float const *gamma_in,ulong  gamma_in_offset,
                              __global float *x_factor,ulong  x_factor_offset,
                              __global float *dy_factor,ulong  dy_factor_offset,
                              __global float *offset,ulong  offset_offset)
{
    int i = get_global_id(0);
    if(i >= N)
        return;
    x_factor += x_factor_offset;
    dy_factor += dy_factor_offset;
    offset += offset_offset; 
    mean += mean_offset;
    varrstd  += varrstd_offset;
    if(gamma_in)
        gamma_in += gamma_in_offset;
    dyx_sum += dyx_sum_offset;
    dy_sum  += dy_sum_offset;

    float one_by_M = 1.0f / M;
    float rsqrtsig;
    if(eps < 0)
        rsqrtsig = varrstd[i];
    else
        rsqrtsig = 1.0f / sqrt(varrstd[i] + eps);

    float gamma=1.0f;
    if(gamma_in)
        gamma = gamma_in[i];
    float mu = mean[i];
    float dys = dy_sum[i];
    float dsig = -0.5 * gamma * (dyx_sum[i] - mu * dys) * (rsqrtsig * rsqrtsig * rsqrtsig);
    float gamma_div_sigsqrt = gamma * rsqrtsig;
    float dmu = -dys * gamma_div_sigsqrt;
    float F_dy = gamma_div_sigsqrt;
    float F_x  = 2*dsig * one_by_M;
    float B = one_by_M * (dmu - dsig * 2 * mu);

    dy_factor[i] = F_dy;
    x_factor[i] = F_x;
    offset[i] = B;
}
