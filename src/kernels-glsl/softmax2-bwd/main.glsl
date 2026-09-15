#version 450
#include "../common/defs.glsl"
layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint localSizeX = 1;
#include "../common/shape.glsl"

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
	layout(binding = 0, std430) writeonly buffer x0_grad_buf { typeof_x0 x0_grad_data[]; };
	layout(binding = 1, std430) readonly buffer y0_buf { typeof_y0 y0_data[]; };
	layout(binding = 2, std430) readonly buffer y0_grad_buf { typeof_y0 y0_grad_data[]; };
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_grad_offset;
	Shape x0_grad_strides;
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	Shape y0_strides;
	
	#if USE_BDA
		// y0_grad_data
	#endif
	uint y0_grad_offset;
	Shape y0_grad_strides;

	Shape xShape;
	Shape reduceDims;
};

// This should be equal to the amount of reduce elements
shared acctype yShmem[SHMEM_SIZE];

// gave it silly name to prevent kernel source preprocessing for now
void main()
{
	// also adapted from here
	// https://adityaagrawal.net/blog/deep_learning/bprop_softmax
	
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
	
	// store these for later
	precise acctype y0_cache[WPT];
	precise acctype y0_grad_cache[WPT];
	
	// This whole segment is for calculating the partial sum portion
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
		// load y and y grad, do the thingy
		uint y0_idx = y0_offset + getStridedIndexFromPos(xPos, y0_strides, DIMS);
		precise acctype y0 = acctype(y0_data[y0_idx]);
		y0_cache[wptElem] = y0;
		uint y0_grad_idx = y0_grad_offset + getStridedIndexFromPos(xPos, y0_grad_strides, DIMS);
		precise acctype y0_grad = acctype(y0_grad_data[y0_grad_idx]);
		y0_grad_cache[wptElem] = y0_grad;
		if (SOFTMAX_ROUTINE == ROUTINE_SOFTMAX)
			yReduce += y0_grad*y0;
		else
			yReduce += y0_grad;
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
			// how does this work for the backward? I don't think it works like this.
			if (SOFTMAX_ROUTINE == ROUTINE_LOG_SOFTMAX)
				yReduce = log(yReduce);
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
				yReduce = log(yReduce);
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
		if (i >= NUM_REDUCE_ELEMS) continue;
		
		// adjust position to point to the specific element being iterated on
		Shape reduceOpPos = getPos(i, reduceShape, NUM_REDUCE_DIMS);
		[[unroll]]
		for (uint j = 0; j < NUM_REDUCE_DIMS; j += 1) xPos.s[reduceDims.s[j]] = reduceOpPos.s[j];
		
		if ( !posValid(xShape, xPos, DIMS) ) continue;
		
		precise acctype x0_grad;
		if (SOFTMAX_ROUTINE == ROUTINE_SOFTMAX)
			x0_grad = acctype(-1.0)*y0_cache[wptElem]*(yReduce - y0_grad_cache[wptElem]);
		else
			x0_grad = y0_grad_cache[wptElem] - exp(y0_cache[wptElem])*yReduce;
			
		// load x value, this time use the complete denominator to compute the softmax.
		uint x0_grad_idx = x0_grad_offset + getStridedIndexFromPos(xPos, x0_grad_strides, DIMS);
		x0_grad_data[x0_grad_idx] = typeof_x0(x0_grad);
	}
}
