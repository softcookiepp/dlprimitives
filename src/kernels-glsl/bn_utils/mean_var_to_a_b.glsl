#version 450


void mean_var_to_a_b(int N,float eps,
                     __global float const * mean,uint  mean_offset,
                     __global float const * var, uint  var_offset,
                     __global float *a,uint  a_offset,
                     __global float *b,uint  b_offset)
void main()
{
    uint pos = get_global_id(0);
    if(pos >= N)
        return;
    float scale = 1.0f / sqrt(var[pos + var_offset] + eps);
    float offset  = - mean[pos + mean_offset] * scale;
    a[pos + a_offset] = scale;
    b[pos + b_offset] = offset;
}
