#version 450
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x_buf { float x[]; };
	layout(binding = 1, std430) buffer y_buf { float y[]; };
	layout(binding = 2, std430) readonly buffer A_buf { float A[]; };
	layout(binding = 3, std430) readonly buffer B_buf { float B[]; };
#endif

layout(push_constant, std430) uniform forward
{
	uint batches; uint channels; uint HW;
	#if USE_BDA
		__global float const *x;
	#endif
	uint  x_offset;
	#if USE_BDA
		__global float *y;      
	#endif
	uint  y_offset;
	#if USE_BDA
		__global float const *A;
	#endif
	uint A_offset;
	#if USE_BDA
		__global float const *B;
	#endif
	uint B_offset;
};
             
void main()
{
    uint b  = get_global_id(DIM_B);
    uint f  = get_global_id(DIM_F);
    uint rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    uint pos = (b * channels + f) * HW + rc;
    y[y_offset + pos] = x[x_offset + pos] * A[A_offset + f] + B[B_offset + f];
}
