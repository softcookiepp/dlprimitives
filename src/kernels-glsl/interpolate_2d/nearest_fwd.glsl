#version 450
#include "../common/defs.glsl"
#include "../common/atomic.glsl"
#include "../common/workgroup.glsl"
#include "common_functions.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x_buf { dtype x[]; };
	layout(binding = 1, std430) buffer y_buf { dtype y[]; };
#endif

layout(push_constant, std430) uniform nearest_fwd
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
	__global const dtype* x;
#endif
	uint x_offset;
#if USE_BDA
	__global dtype* y;
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

    uint src_r = get_src_pos(r,scale_y,srcH,offset);
    uint src_c = get_src_pos(c,scale_x,srcW,offset);
    x_ += src_r * srcW + src_c;
    y_ += r * tgtW + c;

    for(uint n=n0;n<n1;n++) {
        y[y_] = x[x_];
        x_ += step_src;
        y_ += step_tgt;
    }
}
