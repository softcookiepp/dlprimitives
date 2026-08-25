#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#define NUM_WEIGHTS_MAX 64 // we may be able to reduce this later
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;

#ifndef typeof_x0
	#define typeof_x0 dtype
#endif
#ifndef typeof_y0
	#define typeof_y0 dtype
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
	layout(binding = 1, std430) buffer y0_buf { typeof_y0 y0_data[]; };
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_offset;
	uint x0_inc;
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	uint y0_inc;
	
	uint total; // total elements
	
	float w[NUM_WEIGHTS];
};

#include "unary-unary-templates.glsl"

void main()
{
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= total) return;
	uint x0_pos = x0_offset + gid*x0_inc;
	typeof_x0 x0 = x0_data[x0_pos];
	
	uint y0_pos = y0_offset + gid*y0_inc;
	y0_data[y0_pos] = unary_unary_function(x0);
}
