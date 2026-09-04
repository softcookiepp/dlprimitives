#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;
layout(constant_id = 5) const uint DIMS = DIMS_MAX;
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
	layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
	#if X_ARITY > 1
		layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
	#endif
	#if X_ARITY > 2
		layout(binding = 2, std430) readonly buffer x2_buf { typeof_x2 x2_data[]; };
	#endif
	layout(binding = X_ARITY, std430) buffer y0_buf { typeof_y0 y0_data[]; };
	#if Y_ARITY > 1
		layout(binding = X_ARITY + 1, std430) buffer y1_buf { typeof_y1 y1_data[]; };
	#endif
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_offset;
	Shape x0_strides;
	
	#if X_ARITY > 1
		#if USE_BDA
			// x1_data
		#endif
		uint x1_offset;
		Shape x1_strides;
	#endif
	#if X_ARITY > 2
		#if USE_BDA
			// x2_data
		#endif
		uint x2_offset;
		Shape x2_strides;
	#endif
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	Shape y0_strides;
	#if Y_ARITY > 1
		#if USE_BDA
			// y1_data
		#endif
		uint y1_offset;
		Shape y1_strides;
	#endif
	Shape shape;
	W_ARGS wArgs;
};

void pointwise_strided_impl()
{
	// determine position, exit if out of bounds
	Shape pos = getPosFromTriIndex(shape, DIMS);
	if (!posValid(shape, pos, DIMS)) return;
	
	// load x values
	X_IN xArgs;
	uint x0_idx = x0_offset + getStridedIndexFromPos(pos, x0_strides, DIMS);
	xArgs.data[0] = acctype(x0_data[x0_idx]);
	#if X_ARITY > 1
		uint x1_idx = x1_offset + getStridedIndexFromPos(pos, x1_strides, DIMS);
		xArgs.data[1] = acctype(x1_data[x1_idx]);
	#endif
	#if X_ARITY > 2
		uint x2_idx = x2_offset + getStridedIndexFromPos(pos, x2_strides, DIMS);
		xArgs.data[2] = acctype(x2_data[x2_idx]);
	#endif
	
	// calculate everything
	Y_OUT y = pointwise_function(pos, gl_GlobalInvocationID.x, X_ARITY, Y_ARITY, xArgs, wArgs);
	
	// store y values
	uint y0_idx = y0_offset + getStridedIndexFromPos(pos, y0_strides, DIMS);
	y0_data[y0_idx] = typeof_y0(y.data[0]);
	#if Y_ARITY > 1
		uint y1_idx = y1_offset + getStridedIndexFromPos(pos, y1_strides, DIMS);
		y1_data[y1_idx] = typeof_y1(y.data[1]);
	#endif
}
