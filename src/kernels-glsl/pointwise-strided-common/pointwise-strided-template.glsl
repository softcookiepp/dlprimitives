#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;
#if 0
	layout(constant_id = 5) const uint DIMS = DIMS_MAX;
#endif

#include "../pointwise-common/pointwise-routines.glsl"

#ifndef X_ARITY
	#define X_ARITY 1
#endif
#if X_ARITY > X_ARITY_MAX
	#error("X_ARITY too big")
#endif

#ifndef Y_ARITY
	#define Y_ARITY 1
#endif
#if Y_ARITY > Y_ARITY_MAX
	#error("Y_ARITY too big")
#endif

// Even if x1 isn't used, defining type won't hurt anything.
// I don't want too many goofy conditions
#ifndef typeof_x0
	#define typeof_x0 dtype
#endif
#ifndef typeof_x1
	#define typeof_x1 dtype
#endif
#ifndef typeof_x2
	#define typeof_x2 dtype
#endif

#ifndef typeof_y0
	#define typeof_y0 dtype
#endif
#ifndef typeof_y1
	#define typeof_y1 dtype
#endif

#if USE_BDA == 0
	#if X_ARITY == 1
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) buffer y0_buf { typeof_y0 y0_data[]; };
		#if Y_ARITY > 1
			layout(binding = 2, std430) buffer y1_buf { typeof_y1 y1_data[]; };
		#endif
	#elif X_ARITY == 2
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
		layout(binding = 2, std430) buffer y0_buf { typeof_y0 y0_data[]; };
		#if Y_ARITY > 1
			layout(binding = 3, std430) buffer y1_buf { typeof_y1 y1_data[]; };
		#endif
	#elif X_ARITY == 3
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
		layout(binding = 2, std430) readonly buffer x2_buf { typeof_x2 x2_data[]; };
		layout(binding = X_ARITY, std430) buffer y0_buf { typeof_y0 y0_data[]; };
	#else
		#error "too big"
	#endif
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_offset;
	#if 0
		Shape x0_strides;
	#else
		uint x0_inc;
	#endif
	
	#if X_ARITY > 1
		#if USE_BDA
			// x1_data
		#endif
		uint x1_offset;
		#if 0
			Shape x1_strides;
		#else
			uint x1_inc;
		#endif
	#endif
	#if X_ARITY > 2
		#if USE_BDA
			// x2_data
		#endif
		uint x2_offset;
		#if 0
			Shape x2_strides;
		#else
			uint x2_inc;
		#endif
	#endif
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	#if 0
		Shape y0_strides;
	#else
		uint y0_inc;
	#endif
	#if Y_ARITY > 1
		#if USE_BDA
			// y1_data
		#endif
		uint y1_offset;
		#if 0
			Shape y1_strides;
		#else
			uint y1_inc;
		#endif
	#endif
	
	#if 0
		Shape broadcast_strides;
	#else
		uint total; // total elements
	#endif
	W_ARGS wArgs;
};

void pointwise_strided_impl()
{
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= total) return;
	
	#if 0
		// get position
		Shape pos = get_pos(gid, broadcast_shape);
		UNROLL(DIMS)
		for (uint i = 0; i < DIMS; i += 1)
		{
			if (pos[i] >= broadcast_shape[i]) return;
		}
	#endif
	
	X_IN xArgs;
	
	#if 0
	#else
		uint x0_idx = x0_offset + gid*x0_inc;
	#endif
	typeof_x0 x0 = x0_data[x0_idx];
	xArgs.data[0] = acctype(x0);
	#if X_ARITY > 1
		#if 0
		#else
			uint x1_idx = x1_offset + gid*x1_inc;
		#endif
		typeof_x1 x1 = x1_data[x1_idx];
		xArgs.data[1] = acctype(x1);
	#endif
	#if X_ARITY > 2
		#if 0
		#else
			uint x2_idx = x2_offset + gid*x2_inc;
		#endif
		typeof_x2 x2 = x2_data[x2_idx];
		xArgs.data[2] = acctype(x2);
	#endif
	
	#if 0
	#else
		uint y0_idx = y0_offset + gid*y0_inc;
	#endif
	
	Y_OUT y = pointwise_function(gid, X_ARITY, Y_ARITY, xArgs, wArgs);
	#if Y_ARITY == 1
		y0_data[y0_idx] = typeof_y0(y.data[0]);
	#elif Y_ARITY == 2
		#if 0
		#else
			uint y1_idx = y1_offset + gid*y1_inc;
		#endif
		y0_data[y0_idx] = typeof_y0(y.data[0]);
		y1_data[y1_idx] = typeof_y1(y.data[1]);
	#endif
}
