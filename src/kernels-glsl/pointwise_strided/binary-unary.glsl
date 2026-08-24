#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x0_buf { dtype x0[]; };
	layout(binding = 0, std430) readonly buffer x1_buf { dtype x1[]; };
	layout(binding = 1, std430) buffer y0_buf { dtype y0[]; };
#endif



