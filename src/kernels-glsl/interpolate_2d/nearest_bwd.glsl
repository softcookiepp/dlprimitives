#version 450
#include "../common/defs.glsl"
#include "../common/atomic.glsl"
#include "../common/workgroup.glsl"
#include "common_functions.glsl"




layout(push_constant, std430) uniform nearest_bwd
{
	uint N;
	uint items_per_thread;
	uint srcH;
	uint srcW;
	uint tgtH;
	uint tgtW;
	float scale_y;
	float scale_x;
	float offset;
#if USE_BDA
	__global dtype* dx;
#endif
	uint dx_offset;
#if USE_BDA
	__global dtype const * dy;
#endif
	uint dy_offset;
	float beta;
};
        
void main()
{
    uint n0 = get_global_id(2) * items_per_thread;
    uint n1 = min(n0 + items_per_thread,N);
    uint src_r = get_global_id(0);
    uint src_c = get_global_id(1);
    if(n0 >= N || src_r>= srcH || src_c>= srcW)
        return;

    uint step_src = srcW*srcH;
    uint step_tgt = tgtW*tgtH;
    dx += dx_offset + n0 * step_src;
    dy += dy_offset + n0 * step_tgt;

    uint tgt_r0 = get_tgt_pos(src_r,  scale_y, tgtH, offset);
    uint tgt_r1 = get_tgt_pos(src_r+1,scale_y, tgtH, offset);
    uint tgt_c0 = get_tgt_pos(src_c,  scale_x, tgtW, offset);
    uint tgt_c1 = get_tgt_pos(src_c+1,scale_x, tgtW, offset);
   
    dx += src_r * srcW + src_c;

    for(uint n=n0;n<n1;n++) {
        dtype grad = 0;
        for(uint r=tgt_r0;r<tgt_r1;r++) {
            for(uint c=tgt_c0;c<tgt_c1;c++) {
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
