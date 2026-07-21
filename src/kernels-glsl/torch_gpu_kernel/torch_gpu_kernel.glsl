#version 450
#include "../common/defs.glsl"

#define INDEXED_FILL 0

// TODO: change to whatever torch says the size is
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint nt = 1;
layout(constant_id = 1) const uint vt = 1;
layout(constant_id 2) const uint FUNCTION_TYPE = 0;

#ifndef KERNEL_ARITY
	#error("Ambiguous number of arguments, as KERNEL_ARITY was not defined")
#else
	#error("NOT IMPLEMENTED")
#endif

void f(uint idx)
{
#if FUNCTION_TYPE == INDEXED_FILL
	
#else
	#error("Unknown function type")
#endif
}

// this is what needs to be ported
__global__ void elementwise_kernel(uint N, func_t f)
{
	uint tid = threadIdx.x;
	uint nv = nt * vt;
	uint idx = nv * blockIdx.x + tid;
	UNROLL(vt)
	for (uint i = 0; i < vt; i++)
	{
		if (idx < N)
		{
			f(idx);
			idx += nt;
		}
	}
}
