#version 450
#include "common/defs.glsl"
layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint localSizeX = 1;
#include "common/shape.glsl"

#define ROUTINE_SOFTMAX 0
#define ROUTINE_LOG_SOFTMAX 1

layout(constant_id = 1) const uint DIMS = DIMS_MAX;
layout(constant_id = 2) const uint NUM_REDUCE_DIMS = DIMS_MAX;
layout(constant_id = 3) const uint NUM_REDUCE_ELEMS = 1024;
layout(constant_id = 4) const uint WPT = 2;
layout(constant_id = 5) const uint SHMEM_SIZE = 1024;
layout(constant_id = 6) const uint SOFTMAX_ROUTINE = ROUTINE_SOFTMAX; // either softmax or log softmax

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
	Shape x0_strides;
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	Shape y0_strides;

	Shape xShape;
	Shape reduceDims;
};

// This should be equal to the amount of reduce elements
shared acctype yShmem[SHMEM_SIZE];

void main()
{
	// This kernel was adapted from the pointwise broadcast reduce kernel.
	// Given their similarity, it made sense to do so.
	// It definitely needs cleaning up though.
	
	// determine position, exit if out of bounds
	// In this kernel, yShape is the one
	// To save on push constant space, y shape is determined form x shape and reduce dims
	Shape yShape;
	[[unroll]]
	for (uint i = 0; i < DIMS; i += 1)
		yShape.s[i] = xShape.s[i];
	[[unroll]]
	for (uint i = 0; i < NUM_REDUCE_DIMS; i += 1)
		yShape.s[reduceDims.s[i]] = 1;
		
	Shape yPos = getPosFromTriIndex(gl_WorkGroupID, yShape, DIMS);
	if (!posValid(yShape, yPos, DIMS)) return;

	Shape xPos = yPos;
	precise acctype yReduce = acctype(0.0);
	
	// initialize shared memory
	#if USE_SUBGROUP_ARITHMETIC
		if (gl_SubgroupInvocationID == 0)
			yShmem[gl_SubgroupID] = yReduce;
	#else
		yShmem[gl_LocalInvocationID.x] = yReduce;
	#endif
	barrier();
	
	if (gl_LocalInvocationID.x*WPT > NUM_REDUCE_ELEMS) return;
	
	// Also get reduceShape, to avoid having too many push constants
	Shape reduceShape;
	[[unroll]]
	for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1) reduceShape.s[j] = xShape.s[reduceDims.s[j]];
	
	// for x0 values so they don't have to be reloaded later
	precise acctype x0_values[WPT];
	// This whole segment is for calculating the denominator of the softmax function.
	[[unroll]]
	for (uint wptElem = 0; wptElem < WPT; wptElem += 1)
	{
		uint i = gl_LocalInvocationID.x*WPT + wptElem;
		
		// Ensure we don't accidentally go over the number of reduce elems.
		// For the naive implementation where the reduction is just an iteration, this doesn't matter.
		// But it will for later implementations.
		if (i >= NUM_REDUCE_ELEMS) continue;
		
		// adjust position to point to the specific element being iterated on
		Shape reduceOpPos = getPos(i, reduceShape, NUM_REDUCE_DIMS);
		[[unroll]]
		for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1) xPos.s[reduceDims.s[j]] = reduceOpPos.s[j];
		
		if ( !posValid(xShape, xPos, DIMS) ) continue;
		// load x value, add it to denominator
		uint x0_idx = x0_offset + getStridedIndexFromPos(xPos, x0_strides, DIMS);
		precise acctype x0 = acctype(x0_data[x0_idx]);
		x0_values[wptElem] = x0; // store for later use
		yReduce += exp(x0);
	}
	
	
	#if USE_SUBGROUP_ARITHMETIC
		// Reduce with subgroup arithmetic
		yReduce = subgroupAdd(yReduce);
		subgroupBarrier();
		
		// Now with softmax, this gets tricky.
		// Store each subgroup-accumulated partial sum in local memory
		if (gl_SubgroupInvocationID == 0)
			yShmem[gl_SubgroupID] = yReduce;
		barrier();
		// wait, can't we just use all subgroups for this?
		{
			// re-initialize yReduce yet again
			yReduce = acctype(0.0);
			// iterate over each subgroup-compute partial sum and add them together
			[[unroll]]
			for (uint i = 0; i < gl_NumSubgroups; i += 1)
			{
				yReduce += yShmem[i];
			}
			// take the log of it if necessary
			if (SOFTMAX_ROUTINE == ROUTINE_LOG_SOFTMAX)
				yReduce = clippedNaturalLog(yReduce);
		}
		// at this point, yReduce should be the fully computed softmax denominator.
		
	#else
		// No subgroup support, fall back to storing all partial sums in local memory and adding them.
		// I was too stupid to figure out a better way to do this.
		// If anyone else knows how, I would very much appreciate it!
		
		yShmem[gl_LocalInvocationID.x] = yReduce;
		barrier();
		if (gl_LocalInvocationID.x == 0)
		{
			// compute the denominator
			yReduce = acctype(0.0);
			
			[[unroll]]
			for (uint i = 0; i < SHMEM_SIZE; i += 1)
			{
				yReduce += yShmem[i];
			}
			if (SOFTMAX_ROUTINE == ROUTINE_LOG_SOFTMAX)
				yReduce = clippedNaturalLog(yReduce);
			[[unroll]]
			for (uint i = 0; i < SHMEM_SIZE; i += 1)
			{
				yShmem[i] = yReduce;
			}
		}
		barrier();
		// now all threads load it back in!
		yReduce = yShmem[gl_LocalInvocationID.x];
	#endif
	
	// now we gotta do another one of these.
	[[unroll]]
	for (uint wptElem = 0; wptElem < WPT; wptElem += 1)
	{
		uint i = gl_LocalInvocationID.x*WPT + wptElem;
		
		// Ensure we don't accidentally go over the number of reduce elems.
		// For the naive implementation where the reduction is just an iteration, this doesn't matter.
		// But it will for later implementations.
		if (i >= NUM_REDUCE_ELEMS) continue;
		
		// adjust position to point to the specific element being iterated on
		Shape reduceOpPos = getPos(i, reduceShape, NUM_REDUCE_DIMS);
		[[unroll]]
		for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1) xPos.s[reduceDims.s[j]] = reduceOpPos.s[j];
		
		if ( !posValid(xShape, xPos, DIMS) ) continue;
		// load x value, this time use the complete denominator to compute the softmax.
		#if 1
			precise acctype x0 = acctype(x0_values[wptElem]);
		#else
			uint x0_idx = x0_offset + getStridedIndexFromPos(xPos, x0_strides, DIMS);
			precise acctype x0 = acctype(x0_data[x0_idx]);
		#endif
		if (SOFTMAX_ROUTINE == ROUTINE_SOFTMAX)
			x0 = exp(x0)/yReduce;
		else
			x0 = x0 - yReduce;
		
		// compute + store results
		uint y0_idx = y0_offset + getStridedIndexFromPos(xPos, y0_strides, DIMS);
		y0_data[y0_idx] = typeof_y0(x0);
	}
}
