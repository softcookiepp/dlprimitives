#version 450
__kernel
void backward_test(int batches,int channels,int HW,
             __global float *dx,ulong  dx_offset,
             __global float const *dy,ulong  dy_offset,
             __global float const *a,ulong  a_offset,
             float factor)
{
    int b  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    dx+=dx_offset;
    dy+=dy_offset;
    a+=a_offset;
    int pos = (b * channels + f) * HW + rc;
    float val = dy[pos] * a[f];
    if(factor == 0)
        dx[pos] = val;
    else
        dx[pos] = dx[pos]*factor + val;
}
