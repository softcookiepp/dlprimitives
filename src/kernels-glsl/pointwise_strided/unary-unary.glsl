#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
layout(constant_id = 3) const uint NUM_WEIGHTS = 1;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x0_buf { dtype x0_data[]; };
	layout(binding = 1, std430) buffer y0_buf { dtype y0_data[]; };
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
	
	float w[NUM_WEIGHTS];
};

#include "unary-unary-template.glsl"
