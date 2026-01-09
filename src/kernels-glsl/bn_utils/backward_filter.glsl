#version 450

__kernel
void backward_filter(int N,
                    __global float const *mean,ulong  mean_offset,
                    __global float const *var, ulong  var_offset,
                    __global float const *dy_sum, ulong  dy_sum_offset,
                    __global float const *dyx_sum,ulong  dyx_sum_offset,
                    __global float *dgamma,ulong  dgamma_offset,
                    __global float *dbeta,ulong  dbeta_offset,
                    float eps,
                    float factor_gamma,
                    float factor_beta)
{
    int i=get_global_id(0);
    if(i >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    dy_sum += dy_sum_offset;
    dyx_sum += dyx_sum_offset;

    float dys = dy_sum[i];

    if(dgamma) {
        dgamma += dgamma_offset;
        float dG = (dyx_sum[i] - mean[i]*dys) / sqrt(var[i] + eps); 
        if(factor_gamma == 0)
            dgamma[i] = dG;
        else
            dgamma[i] = dgamma[i]*factor_gamma + dG;
    }
    
    if(dbeta) {
        dbeta += dbeta_offset;
        if(factor_beta == 0)
            dbeta[i] = dys;
        else
            dbeta[i] = dbeta[i] * factor_beta + dys;
    }
}
