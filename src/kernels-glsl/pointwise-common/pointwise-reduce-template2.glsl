#include "../common/defs.glsl"

// these will be adjusted as time goes on
#define X_ARITY_MAX 4
#define Y_ARITY_MAX 2

layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint localSizeX = 1;
#include "../common/shape.glsl"
#define NUM_WEIGHTS_MAX 8
layout(constant_id = 1) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 2) const uint DIMS = DIMS_MAX;
layout(constant_id = 3) const uint NUM_REDUCE_DIMS = DIMS_MAX;
layout(constant_id = 4) const uint NUM_REDUCE_ELEMS = 1024;
layout(constant_id = 5) const uint WPT = 2;
layout(constant_id = 6) const uint SHMEM_SIZE = 1024;
layout(constant_id = 7) const uint POINTWISE_ROUTINE = 0;
layout(constant_id = 8) const uint REDUCE_ROUTINE = 0;
layout(constant_id = 9) const uint X_ARITY = X_ARITY_MAX;
layout(constant_id = 10) const uint Y_ARITY = Y_ARITY_MAX;

#define USE_SPEC_FOR_STRIDES 0

#if USE_SPEC_FOR_STRIDES
	// This is the only way to bypass the push constant limit without using uniform buffers.
	// Uniform buffers require extra command buffer recording to upload the arguments,
	// but too many specialization constant permutations will mean excess pipeline overhead.
	// It remains to be seen which one is worse.
	layout(constant_id = 11) const uint X0_S0 = 0;
	layout(constant_id = 12) const uint X0_S1 = 0;
	layout(constant_id = 13) const uint X0_S2 = 0;
	layout(constant_id = 14) const uint X0_S3 = 0;
	layout(constant_id = 15) const uint X0_S4 = 0;
	layout(constant_id = 16) const uint X0_S5 = 0;
	layout(constant_id = 17) const uint X0_S6 = 0;
	layout(constant_id = 18) const uint X0_S7 = 0;
	layout(constant_id = 19) const uint X0_S8 = 0;
	const Shape x0_strides(uint[DIMS_MAX](
		X0_S0,
		X0_S1,
		X0_S2,
		X0_S3,
		X0_S4,
		X0_S5,
		X0_S6,
		X0_S7
	));

	layout(constant_id = 20) const uint X1_S0 = 0;
	layout(constant_id = 21) const uint X1_S1 = 0;
	layout(constant_id = 22) const uint X1_S2 = 0;
	layout(constant_id = 23) const uint X1_S3 = 0;
	layout(constant_id = 24) const uint X1_S4 = 0;
	layout(constant_id = 25) const uint X1_S5 = 0;
	layout(constant_id = 26) const uint X1_S6 = 0;
	layout(constant_id = 27) const uint X1_S7 = 0;
	layout(constant_id = 28) const uint X1_S8 = 0;
	const Shape x1_strides(uint[DIMS_MAX](
		X1_S0,
		X1_S1,
		X1_S2,
		X1_S3,
		X1_S4,
		X1_S5,
		X1_S6,
		X1_S7
	));

	layout(constant_id = 29) const uint X2_S0 = 0;
	layout(constant_id = 30) const uint X2_S1 = 0;
	layout(constant_id = 31) const uint X2_S2 = 0;
	layout(constant_id = 32) const uint X2_S3 = 0;
	layout(constant_id = 33) const uint X2_S4 = 0;
	layout(constant_id = 34) const uint X2_S5 = 0;
	layout(constant_id = 35) const uint X2_S6 = 0;
	layout(constant_id = 36) const uint X2_S7 = 0;
	layout(constant_id = 37) const uint X2_S8 = 0;
	const Shape x2_strides(uint[DIMS_MAX](
		X2_S0,
		X2_S1,
		X2_S2,
		X2_S3,
		X2_S4,
		X2_S5,
		X2_S6,
		X2_S7
	));

	layout(constant_id = 38) const uint X3_S0 = 0;
	layout(constant_id = 39) const uint X3_S1 = 0;
	layout(constant_id = 40) const uint X3_S2 = 0;
	layout(constant_id = 41) const uint X3_S3 = 0;
	layout(constant_id = 42) const uint X3_S4 = 0;
	layout(constant_id = 43) const uint X3_S5 = 0;
	layout(constant_id = 44) const uint X3_S6 = 0;
	layout(constant_id = 45) const uint X3_S7 = 0;
	layout(constant_id = 46) const uint X3_S8 = 0;
	const Shape x3_strides(uint[DIMS_MAX](
		X3_S0,
		X3_S1,
		X3_S2,
		X3_S3,
		X3_S4,
		X3_S5,
		X3_S6,
		X3_S7
	));

	layout(constant_id = 47) const uint Y0_S0 = 0;
	layout(constant_id = 48) const uint Y0_S1 = 0;
	layout(constant_id = 49) const uint Y0_S2 = 0;
	layout(constant_id = 50) const uint Y0_S3 = 0;
	layout(constant_id = 51) const uint Y0_S4 = 0;
	layout(constant_id = 52) const uint Y0_S5 = 0;
	layout(constant_id = 53) const uint Y0_S6 = 0;
	layout(constant_id = 54) const uint Y0_S7 = 0;
	layout(constant_id = 55) const uint Y0_S8 = 0;
	const Shape y0_strides(uint[DIMS_MAX](
		Y0_S0,
		Y0_S1,
		Y0_S2,
		Y0_S3,
		Y0_S4,
		Y0_S5,
		Y0_S6,
		Y0_S7
	));

	layout(constant_id = 11) const uint Y1_S0 = 0;
	layout(constant_id = 11) const uint Y1_S1 = 0;
	layout(constant_id = 11) const uint Y1_S2 = 0;
	layout(constant_id = 11) const uint Y1_S3 = 0;
	layout(constant_id = 11) const uint Y1_S4 = 0;
	layout(constant_id = 11) const uint Y1_S5 = 0;
	layout(constant_id = 11) const uint Y1_S6 = 0;
	layout(constant_id = 11) const uint Y1_S7 = 0;
	layout(constant_id = 11) const uint Y1_S8 = 0;
	const Shape y1_strides(uint[DIMS_MAX](
		Y1_S0,
		Y1_S1,
		Y1_S2,
		Y1_S3,
		Y1_S4,
		Y1_S5,
		Y1_S6,
		Y1_S7
	));
#endif

#include "../pointwise-common/pointwise-routines.glsl"

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
#ifndef typeof_x3
	#define typeof_x3 dtype
#endif

#ifndef typeof_y0
	#define typeof_y0 dtype
#endif
#ifndef typeof_y1
	#define typeof_y1 dtype
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
	layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
	layout(binding = 2, std430) readonly buffer x2_buf { typeof_x2 x2_data[]; };
	layout(binding = 3, std430) readonly buffer x3_buf { typeof_x3 x3_data[]; };
	
	layout(binding = 4, std430) buffer y0_buf { typeof_y0 y0_data[]; };
	layout(binding = 5, std430) buffer y1_buf { typeof_y1 y1_data[]; };
	#if USE_SPEC_FOR_STRIDES == 0
		layout(binding = 6) uniform uni0_buf
		{
			#if 1
				// because std140 is incredibly stupid
				uint x0s0;
				uint x0s1;
				uint x0s2;
				uint x0s3;
				uint x0s4;
				uint x0s5;
				uint x0s6;
				uint x0s7;
				
				uint x1s0;
				uint x1s1;
				uint x1s2;
				uint x1s3;
				uint x1s4;
				uint x1s5;
				uint x1s6;
				uint x1s7;
				
				uint x2s0;
				uint x2s1;
				uint x2s2;
				uint x2s3;
				uint x2s4;
				uint x2s5;
				uint x2s6;
				uint x2s7;
				
				uint x3s0;
				uint x3s1;
				uint x3s2;
				uint x3s3;
				uint x3s4;
				uint x3s5;
				uint x3s6;
				uint x3s7;
				
				uint y0s0;
				uint y0s1;
				uint y0s2;
				uint y0s3;
				uint y0s4;
				uint y0s5;
				uint y0s6;
				uint y0s7;
				
				uint y1s0;
				uint y1s1;
				uint y1s2;
				uint y1s3;
				uint y1s4;
				uint y1s5;
				uint y1s6;
				uint y1s7;
				
				uint xShape0;
				uint xShape1;
				uint xShape2;
				uint xShape3;
				uint xShape4;
				uint xShape5;
				uint xShape6;
				uint xShape7;
				
				uint r0;
				uint r1;
				uint r2;
				uint r3;
				uint r4;
				uint r5;
				uint r6;
				uint r7;
			#else
				Shape x0_strides;
				Shape x1_strides;
				Shape x2_strides;
				Shape x3_strides;
				Shape y0_strides;
				Shape y1_strides;

				Shape xShape;
				Shape reduceDims;
			#endif
		};
	#endif
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_offset;
	
	#if USE_BDA
		// x1_data
	#endif
	uint x1_offset;

	#if USE_BDA
		// x2_data
	#endif
	uint x2_offset;

	#if USE_BDA
		// x3_data
	#endif
	uint x3_offset;
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;

	#if USE_BDA
		// y1_data
	#endif
	uint y1_offset;

	float yReduceInit[Y_ARITY]; // initial values of y for reduction
	W_ARGS wArgs;
};


// This should be equal to the amount of reduce elements
shared Y_OUT yShmem[SHMEM_SIZE];

void pointwise_reduce_naive_impl()
{	
	#if USE_SPEC_FOR_STRIDES == 0
		// have to make the strides + shapes from the uniform buffer
		Shape x0_strides = Shape(uint[DIMS_MAX](
			x0s0,
			x0s1,
			x0s2,
			x0s3,
			x0s4,
			x0s5,
			x0s6,
			x0s7
		));
		Shape x1_strides = Shape(uint[DIMS_MAX](
			x1s0,
			x1s1,
			x1s2,
			x1s3,
			x1s4,
			x1s5,
			x1s6,
			x1s7
		));
		Shape x2_strides = Shape(uint[DIMS_MAX](
			x2s0,
			x2s1,
			x2s2,
			x2s3,
			x2s4,
			x2s5,
			x2s6,
			x2s7
		));
		Shape x3_strides = Shape(uint[DIMS_MAX](
			x3s0,
			x3s1,
			x3s2,
			x3s3,
			x3s4,
			x3s5,
			x3s6,
			x3s7
		));
		Shape y0_strides = Shape(uint[DIMS_MAX](
			y0s0,
			y0s1,
			y0s2,
			y0s3,
			y0s4,
			y0s5,
			y0s6,
			y0s7
		));
		Shape y1_strides = Shape(uint[DIMS_MAX](
			y1s0,
			y1s1,
			y1s2,
			y1s3,
			y1s4,
			y1s5,
			y1s6,
			y1s7
		));
		Shape xShape = Shape(uint[DIMS_MAX](
			xShape0,
			xShape1,
			xShape2,
			xShape3,
			xShape4,
			xShape5,
			xShape6,
			xShape7
		));
		Shape reduceDims = Shape(uint[DIMS_MAX](
			r0,
			r1,
			r2,
			r3,
			r4,
			r5,
			r6,
			r7
		));
	#endif
	
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

	// Elements are simply loaded sequentially. Why? Because I need something that works before I have something optimal.
	Shape xPos = yPos;
	Y_OUT yReduce;
	[[unroll]]
	for (uint i = 0; i < Y_ARITY; i += 1)
		yReduce.data[i] = acctype(yReduceInit[i]);
	
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
		// load x values
		X_IN xArgs;
		uint x0_idx = x0_offset + getStridedIndexFromPos(xPos, x0_strides, DIMS);
		xArgs.data[0] = acctype(x0_data[x0_idx]);
		if (X_ARITY > 1)
		{
			uint x1_idx = x1_offset + getStridedIndexFromPos(xPos, x1_strides, DIMS);
			xArgs.data[1] = acctype(x1_data[x1_idx]);
		}
		if (X_ARITY > 2)
		{
			uint x2_idx = x2_offset + getStridedIndexFromPos(xPos, x2_strides, DIMS);
			xArgs.data[2] = acctype(x2_data[x2_idx]);
		}
		if (X_ARITY > 3)
		{
			uint x3_idx = x3_offset + getStridedIndexFromPos(xPos, x3_strides, DIMS);
			xArgs.data[3] = acctype(x3_data[x3_idx]);
		}
		
		// compute value, store in shared memory
		Y_OUT yElem = pointwise_function(yPos, x0_idx, X_ARITY, Y_ARITY, xArgs, wArgs, POINTWISE_ROUTINE);
		yReduce = pointwise_reduce_function(yElem, yReduce, Y_ARITY, REDUCE_ROUTINE);
	}
	
	
	#if USE_SUBGROUP_ARITHMETIC
		// Reduce with subgroup arithmetic
		#if 0
			// this is currently very very broken
			Y_OUT ySubgroupReduce;
			[[unroll]]
			for (uint i = 0; i < Y_ARITY; i += 1)
				ySubgroupReduce.data[i] = yReduce.data[i];
			for (uint i = 1; i < gl_SubgroupSize; i += 1)
			{
				if ((gl_LocalInvocationID.x + i)*WPT < NUM_REDUCE_ELEMS)
				{
					Y_OUT ySubgroupSrc;
					[[unroll]]
					for (uint j = 0; j < Y_ARITY; j += 1)
						ySubgroupSrc.data[j] = subgroupShuffleDown(ySubgroupReduce.data[j], i);
				
					yReduce = pointwise_reduce_function(ySubgroupSrc, yReduce, Y_ARITY, REDUCE_ROUTINE);
				}
			}
		#else
			// Most of these ops can just be done by using subgroup arithmetic builtins
			[[unroll]]
			for (uint i = 0; i < Y_ARITY; i += 1)
				yReduce.data[i] = pointwise_subgroup_reduce(yReduce.data[i], REDUCE_ROUTINE);
		#endif
		
		subgroupBarrier();
		if (gl_SubgroupInvocationID > 0) return;
		// Store each subgroup-accumulated partial sum in local memory
		yShmem[gl_SubgroupID] = yReduce;
		barrier();
		
		// And the final phase
		if (gl_LocalInvocationID.x > 0) return;
		
		// re-initialize yReduce yet again
		[[unroll]]
		for (uint i = 0; i < Y_ARITY; i += 1)
			yReduce.data[i] = acctype(yReduceInit[i]);
		// iterate over each subgroup-compute partial sum and add them together
		[[unroll]]
		for (uint i = 0; i < gl_NumSubgroups; i += 1)
		{
			yReduce = pointwise_reduce_function(yShmem[i], yReduce, Y_ARITY, REDUCE_ROUTINE);
		}
	#else
		// No subgroup support, fall back to storing all partial sums in local memory and adding them.
		// I was too stupid to figure out a better way to do this.
		// If anyone else knows how, I would very much appreciate it!
		
		yShmem[gl_LocalInvocationID.x] = yReduce;
		barrier();
		if (gl_LocalInvocationID.x > 0) return;
		[[unroll]]
		for (uint i = 0; i < Y_ARITY; i += 1)
			yReduce.data[i] = acctype(yReduceInit[i]);
		[[unroll]]
		for (uint i = 0; i < SHMEM_SIZE; i += 1)
		{
			yReduce = pointwise_reduce_function(yShmem[i], yReduce, Y_ARITY, REDUCE_ROUTINE);
		}
	#endif
	// store y values
	uint y0_idx = y0_offset + getStridedIndexFromPos(yPos, y0_strides, DIMS);
	y0_data[y0_idx] = typeof_y0(yReduce.data[0]);
	if (Y_ARITY > 1)
	{
		uint y1_idx = y1_offset + getStridedIndexFromPos(yPos, y1_strides, DIMS);
		y1_data[y1_idx] = typeof_y1(yReduce.data[1]);
	}
}
