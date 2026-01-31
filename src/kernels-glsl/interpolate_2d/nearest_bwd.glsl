#version 450
#include "../common/defs.glsl"
#include "../common/atomic.glsl"
#include "../common/workgroup.glsl"
#include "common_functions.glsl"




__kernel 
void nearest_bwd(
        int N,int items_per_thread,
        int srcH,int srcW,
        int tgtH,int tgtW,
        float scale_y,float scale_x,
        float offset,
        __global dtype* dx,ulong dx_offset,
        __global dtype const * dy,ulong dy_offset,float beta)
{
    int n0 = get_global_id(2) * items_per_thread;
    int n1 = min(n0 + items_per_thread,N);
    int src_r = get_global_id(0);
    int src_c = get_global_id(1);
    if(n0 >= N || src_r>= srcH || src_c>= srcW)
        return;

    int step_src = srcW*srcH;
    int step_tgt = tgtW*tgtH;
    dx += dx_offset + n0 * step_src;
    dy += dy_offset + n0 * step_tgt;

    int tgt_r0 = get_tgt_pos(src_r,  scale_y, tgtH, offset);
    int tgt_r1 = get_tgt_pos(src_r+1,scale_y, tgtH, offset);
    int tgt_c0 = get_tgt_pos(src_c,  scale_x, tgtW, offset);
    int tgt_c1 = get_tgt_pos(src_c+1,scale_x, tgtW, offset);
   
    dx += src_r * srcW + src_c;

    for(int n=n0;n<n1;n++) {
        dtype grad = 0;
        for(int r=tgt_r0;r<tgt_r1;r++) {
            for(int c=tgt_c0;c<tgt_c1;c++) {
                grad += dy[r*tgtW+c];
            }
        }
        if(beta == 0)
            *dx = grad;
        else
            *dx = beta * *dx + grad;
        dx += step_src;
        dy += step_tgt;
    }
}
