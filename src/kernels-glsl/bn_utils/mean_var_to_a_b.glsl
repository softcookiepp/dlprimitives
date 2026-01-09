#version 450


void mean_var_to_a_b(int N,float eps,
                     __global float const * mean,ulong  mean_offset,
                     __global float const * var, ulong  var_offset,
                     __global float *a,ulong  a_offset,
                     __global float *b,ulong  b_offset)
void main()
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    a += a_offset;
    b += b_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    float offset  = - mean[pos] * scale;
    a[pos] = scale;
    b[pos] = offset;
}
