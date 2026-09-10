#include "../common/defs.glsl"
layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint localSizeX = 1;
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 1) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 2) const uint POINTWISE_ROUTINE = 0;
layout(constant_id = 3) const uint REDUCE_ROUTINE = 0;
layout(constant_id = 4) const uint DIMS = DIMS_MAX;
layout(constant_id = 5) const uint NUM_REDUCE_DIMS = DIMS_MAX;
layout(constant_id = 6) const uint NUM_REDUCE_ELEMS = 1024;
layout(constant_id = 7) const uint WORK_PER_THREAD = 2;
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
	Shape reduceShape;
	float yReduceInit[Y_ARITY]; // initial values of y for reduction
	W_ARGS wArgs;
};

// This should be equal to the amount of reduce elements
shared Y_OUT yShmem[localSizeX];

void pointwise_reduce_naive_impl()
{
	// determine position, exit if out of bounds
	// In this kernel, yShape is the one
	Shape yPos = getPosFromTriIndex(gl_GlobalInvocationID, yShape, DIMS);
	if (!posValid(yShape, yPos, DIMS)) return;
	
	Shape xPos = yPos;
	
	Y_OUT yTmp[WORK_PER_THREAD];
	uint startIdx = gl_LocalInvocationID.x*WORK_PER_THREAD;
	if (startIdx >= NUM_REDUCE_ELEMS) return; // no point in proceeding if this is the case
	[[unroll]]
	for (uint l = 0; l < WORK_PER_THREAD; l += 1)
	{
		[[unroll]]
		for (uint j = 0; j < Y_ARITY; j += 1)
			yTmp[l].data[j] = yReduceInit[j];
		
		uint xReduceElem = startIdx + l;
		// Ensure we don't accidentally go over the number of reduce elems.
		// For the naive implementation where the reduction is just an iteration, this doesn't matter.
		// But it will for later implementations.
		if (xReduceElem >= NUM_REDUCE_ELEMS) continue;
		
		// adjust position to point to the specific element being iterated on
		Shape reduceOpPos = getPos(xReduceElem, reduceShape, NUM_REDUCE_DIMS);
		[[unroll]]
		for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1)
		{
			xPos.s[reduceDims.s[j]] = reduceOpPos.s[j];
		}
		
		if ( !posValid(xShape, xPos, DIMS) ) continue;
		// load x values
		X_IN xArgs;
		uint x0_idx = x0_offset + getStridedIndexFromPos(xPos, x0_strides, DIMS);
		xArgs.data[0] = acctype(x0_data[x0_idx]);
		#if X_ARITY > 1
			uint x1_idx = x1_offset + getStridedIndexFromPos(xPos, x1_strides, DIMS);
			xArgs.data[1] = acctype(x1_data[x1_idx]);
		#endif
		#if X_ARITY > 2
			uint x2_idx = x2_offset + getStridedIndexFromPos(xPos, x2_strides, DIMS);
			xArgs.data[2] = acctype(x2_data[x2_idx]);
		#endif
		
		// Do the pointwise routine
		yTmp[l] = pointwise_function(yPos, gl_GlobalInvocationID.x, X_ARITY, Y_ARITY, xArgs, wArgs, POINTWISE_ROUTINE);
		
		// TODO: it is possible that different y outputs will require different pointwise operators. Implement this.
		
	}
	
	// Reduce all the WPT elements to a single output (wait I didn't have to split this into another loop. but whatever
	Y_OUT yReduce;
	[[unroll]]
	for (uint i = 0; i < Y_ARITY; i += 1)
		yReduce.data[i] = acctype(yReduceInit[i]);
	
	[[unroll]]
	for (uint wptIdx = 0; wptIdx < WORK_PER_THREAD; wptIdx += 1)
	{
		[[unroll]]
		for (uint j = 0; j < Y_ARITY; j += 1)
		{
			X_IN reduceArgs;
			reduceArgs.data[0] = yTmp[wptIdx].data[j];
			reduceArgs.data[1] = yReduce.data[j];
			yReduce.data[j] = pointwise_function(yPos, gl_GlobalInvocationID.x, 2, 1, reduceArgs, wArgs, REDUCE_ROUTINE).data[0];
		}
	}
	
	// Now to put them all in shared memory and reduce across it.
	yShmem[gl_LocalInvocationID.x] = yReduce;
	barrier();
	
	#if 1
		// For now, start with what we know will work; that is, iterating over the entire thing
		if (gl_LocalInvocationID.x > 0) return;
		
		// re-initialize yReduce
		[[unroll]]
		for (uint i = 0; i < Y_ARITY; i += 1)
			yReduce.data[i] = acctype(yReduceInit[i]);
		
		
		for (uint i = 0; i < localSizeX; i += 1)
		{
			// this again
			[[unroll]]
			for (uint j = 0; j < Y_ARITY; j += 1)
			{
				X_IN reduceArgs;
				reduceArgs.data[0] = yShmem[i].data[j];
				reduceArgs.data[1] = yReduce.data[j];
				yReduce.data[j] = pointwise_function(yPos, gl_GlobalInvocationID.x, 2, 1, reduceArgs, wArgs, REDUCE_ROUTINE).data[0];
			}
		}
	#else
		for (uint s = localSizeX/2; s > 0; s = s >> 1)
		{
			if (gl_LocalInvocationID.x < s)
			{
				uint lidx0 = gl_LocalInvocationID.x;
				uint lidx1 = gl_LocalInvocationID.x + s;
				Y_OUT y0 = yShmem[lidx0];
				Y_OUT y1 = yShmem[lidx1];

				//yShmem[gl_LocalInvocationID.x] = // REDUCE HERE
			}
			barrier();
		}
	#endif
	// store y values
	uint y0_idx = y0_offset + getStridedIndexFromPos(yPos, y0_strides, DIMS);
	y0_data[y0_idx] = typeof_y0(yReduce.data[0]);
}
