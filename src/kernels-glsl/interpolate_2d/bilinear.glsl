#version 450
#include "../common/defs.glsl"
#include "../common/atomic.glsl"
#include "../common/workgroup.glsl"
#include "common_functions.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer x_buf { atomic_dtype x[]; };
	layout(binding = 1, std430) buffer y_buf { dtype y[]; };
#endif

layout(push_constant, std430) uniform bilinear
{
	bool fwd;
	uint N;uint items_per_thread;
	uint srcH;uint srcW;
	uint tgtH;uint tgtW;
	float scale_y;float scale_x;
	bool align_corners;
#if USE_BDA
	__global dtype* restrict x;
#endif
	uint x_offset;
#if USE_BDA
	__global dtype* restrict y;
#endif
	uint y_offset;
};

void main()
{
    uint n0 = get_global_id(2) * items_per_thread;
    uint n1 = min(n0 + items_per_thread,N);
    uint r = get_global_id(0);
    uint c = get_global_id(1);
    if(n0 >= N || r>= tgtH || c>= tgtW)
        return;

    uint step_src = srcW*srcH;
    uint step_tgt = tgtW*tgtH;
    uint x_ = x_offset + n0 * step_src;
    uint y_ = y_offset + n0 * step_tgt;

    float src_r0f = calc_lin_pos(r, scale_y, align_corners);
    uint src_r0 = uint(src_r0f);
    uint dr = (src_r0 < srcH - 1) ? 1 : 0;
    uint src_r1 = src_r0 + dr;
    dtype w_r1 = src_r0f - src_r0;
    dtype w_r0 = 1 - w_r1;

    float src_c0f = calc_lin_pos(c,scale_x,align_corners);
    uint src_c0 = uint(src_c0f);
    uint dc = (src_c0 < srcW - 1) ? 1 : 0;
    uint src_c1 = src_c0 + dc;
    dtype w_c1 = src_c0f - src_c0;
    dtype w_c0 = 1 - w_c1;

    y_ += r * tgtW + c;

	uint x00 = x_ + src_r0 * srcW + src_c0; 
	uint x01 = x_ + src_r0 * srcW + src_c1; 
    uint x10 = x_ + src_r1 * srcW + src_c0; 
    uint x11 = x_ + src_r1 * srcW + src_c1; 

    for(uint n=n0;n<n1;n++) {
        if(fwd) {
            dtype val = 
                w_r0 * (w_c0 * atomic_to_dtype(x[x00]) + w_c1 * atomic_to_dtype(x[x01]) ) +
                w_r1 * (w_c0 * atomic_to_dtype(x[x10]) + w_c1 * atomic_to_dtype(x[x11]) );
            y[y_] = val;
        }
        else {
            dtype val = y[y_];
            atomic_addf(x00, x, w_r0 * w_c0 * val);
            atomic_addf(x01, x, w_r0 * w_c1 * val);
            atomic_addf(x10, x, w_r1 * w_c0 * val);
            atomic_addf(x11, x, w_r1 * w_c1 * val);
        }

        x00 += step_src;
        x01 += step_src;
        x10 += step_src;
        x11 += step_src;
        y_ += step_tgt;
    }
}


