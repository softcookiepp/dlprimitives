#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;
layout(constant_id = 5) const uint REDUCE_ROUTINE = 0;
layout(constant_id = 6) const uint DIMS = DIMS_MAX;
layout(constant_id = 7) const uint NUM_REDUCE_DIMS = DIMS_MAX;
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
	Shape xShape;
	Shape yShape;
	Shape reduceDims;
	float yReduceInit[Y_ARITY]; // initial values of y for reduction
	W_ARGS wArgs;
};

void pointwise_reduce_naive_impl()
{
	// determine position, exit if out of bounds
	// In this kernel, yShape is the one
	Shape yPos = getPosFromTriIndex(gl_GlobalInvocationID, yShape, DIMS);
	if (!posValid(yShape, yPos, DIMS)) return;
	
	// Need to iterate over all possible elements in the reduction shape.
	// also get the shape of the operation
	uint numReduceElems = 1;
	Shape reduceOpSize;
	for (uint i = 0; i < NUM_REDUCE_DIMS; i += 1)
	{
		uint size = xShape.s[reduceDims.s[i]];
		numReduceElems *= size;
		reduceOpSize.s[i] = size;
	}
	
	// Elements are simply loaded sequentially. Why? Because I need something that works before I have something optimal.
	Shape reduceElemPos = yPos;
	acctype yReduce = acctype(yReduceInit[0]);
	for (uint i = 0; i < numReduceElems + 1; i += 1)
	{
		// adjust position to point to the specific element being iterated on
		Shape reduceOpPos = getPos(i, reduceOpSize, NUM_REDUCE_DIMS);
		for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1)
		{
			reduceElemPos.s[reduceDims.s[j]] = reduceOpPos.s[j];
		}
		
		// Ok, that was a bit silly. Now for the actual valuable part.
		
		// load x values
		X_IN xArgs;
		uint x0_idx = x0_offset + getStridedIndexFromPos(reduceElemPos, x0_strides, DIMS);
		xArgs.data[0] = acctype(x0_data[x0_idx]);
		#if X_ARITY > 1
			uint x1_idx = x1_offset + getStridedIndexFromPos(reduceElemPos, x1_strides, DIMS);
			xArgs.data[1] = acctype(x1_data[x1_idx]);
		#endif
		#if X_ARITY > 2
			uint x2_idx = x2_offset + getStridedIndexFromPos(reduceElemPos, x2_strides, DIMS);
			xArgs.data[2] = acctype(x2_data[x2_idx]);
		#endif
		
		// Do the pointwise routine
		Y_OUT yTmp = pointwise_function(yPos, gl_GlobalInvocationID.x, X_ARITY, Y_ARITY, xArgs, wArgs, POINTWISE_ROUTINE);
		
		// For now, Y_ARITY is assumed to be 1. Why? Simply put, anything else will be too complicated, and none of the existing kernels use it
		X_IN reduceArgs;
		reduceArgs.data[0] = yTmp.data[0];
		reduceArgs.data[1] = yReduce;
		yReduce = pointwise_function(yPos, gl_GlobalInvocationID.x, 2, Y_ARITY, reduceArgs, wArgs, REDUCE_ROUTINE).data[0];
	}
	
	// store y values
	uint y0_idx = y0_offset + getStridedIndexFromPos(yPos, y0_strides, DIMS);
	y0_data[y0_idx] = typeof_y0(yReduce);
}
