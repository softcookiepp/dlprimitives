#version 450

__kernel
void backward_data(int batches,int channels,int HW,
             __global float const *x,  ulong  x_offset,
             __global float const *dy, ulong  dy_offset,
             __global float const *fx, ulong  fx_offset,
             __global float const *fdy,ulong  fdy_offset,
             __global float const *b,  ulong  b_offset,
             __global float *dx,       ulong  dx_offset,
             float factor)
{
    int batch  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(batch >= batches || f >= channels || rc >= HW)
        return;
    int pos = (batch * channels + f) * HW + rc;
    float grad =  fx[fx_offset + f] * x[x_offset + pos]  + fdy[fdy_offset + f] * dy[dy_offset + pos] + b[b_offset + f];
    if(factor == 0)
        dx[dx_offset + pos] = grad;
    else
        dx[dx_offset + pos] = dx[dx_offset + pos] * factor + grad;
}
