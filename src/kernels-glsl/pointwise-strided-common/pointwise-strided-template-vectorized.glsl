// None of this is implemented yet. Still needs a lot of refactoring in order to eliminate strides
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;
#ifndef NON_CONTIGUOUS
	#define NON_CONTIGUOUS 1
#endif
#if NON_CONTIGUOUS
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
	#if NON_CONTIGUOUS
		Shape x0_strides;
	#endif
	
	#if X_ARITY > 1
		#if USE_BDA
			// x1_data
		#endif
		uint x1_offset;
		#if NON_CONTIGUOUS
			Shape x1_strides;
		#endif
	#endif
	#if X_ARITY > 2
		#if USE_BDA
			// x2_data
		#endif
		uint x2_offset;
		#if NON_CONTIGUOUS
			Shape x2_strides;
		#endif
	#endif
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	#if NON_CONTIGUOUS
		Shape y0_strides;
	#endif
	#if Y_ARITY > 1
		#if USE_BDA
			// y1_data
		#endif
		uint y1_offset;
		#if NON_CONTIGUOUS
			Shape y1_strides;
		#endif
	#endif
	
	#if NON_CONTIGUOUS
		Shape shape;
	#else
		uint total; // total elements
	#endif
	W_ARGS wArgs;
};

void pointwise_strided_impl()
{
	// determine position, exit if out of bounds
	#if NON_CONTIGUOUS
		Shape pos = getPosFromTriIndex(shape, DIMS);
		if (!posValid(shape, pos, DIMS)) return;
	#else
		uint gid = gl_GlobalInvocationID.x;
		if (gid >= total) return;
	#endif
	
	// load x values
	X_IN xArgs;
	#if NON_CONTIGUOUS
		uint x0_idx = x0_offset + getStridedIndexFromPos(pos, x0_strides, DIMS);
	#else
		uint x0_idx = x0_offset + gid;
	#endif
	xArgs.data[0] = acctype(x0_data[x0_idx]);
	#if X_ARITY > 1
		#if NON_CONTIGUOUS
			uint x1_idx = x1_offset + getStridedIndexFromPos(pos, x1_strides, DIMS);
		#else
			uint x1_idx = x1_offset + gid;
		#endif
		xArgs.data[1] = acctype(x1_data[x1_idx]);
	#endif
	#if X_ARITY > 2
		#if NON_CONTIGUOUS
			uint x2_idx = x2_offset + getStridedIndexFromPos(pos, x2_strides, DIMS);
		#else
			uint x2_idx = x2_offset + gid;
		#endif
		xArgs.data[2] = acctype(x2_data[x2_idx]);
	#endif
	
	// calculate everything
	Y_OUT y = pointwise_function(gl_GlobalInvocationID.x, X_ARITY, Y_ARITY, xArgs, wArgs);
	
	// store y values
	#if NON_CONTIGUOUS
		uint y0_idx = y0_offset + getStridedIndexFromPos(pos, y0_strides, DIMS);
	#else
		uint y0_idx = y0_offset + gid;
	#endif
	y0_data[y0_idx] = typeof_y0(y.data[0]);
	#if Y_ARITY > 1
		#if NON_CONTIGUOUS
			uint y1_idx = y1_offset + getStridedIndexFromPos(pos, y1_strides, DIMS);
		#else
			uint y1_idx = y1_offset + gid;
		#endif
		y1_data[y1_idx] = typeof_y1(y.data[1]);
	#endif
}
