#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#define NUM_WEIGHTS_MAX 64 // we may be able to reduce this later
layout(constant_id = 3) const uint NUM_WEIGHTS = NUM_WEIGHTS_MAX;
layout(constant_id = 4) const uint POINTWISE_ROUTINE = 0;

#define X_ARITY_MAX 2
#ifndef X_ARITY
	#define X_ARITY 1
#endif
#if X_ARITY > X_ARITY_MAX
	#error("X_ARITY too big")
#endif

#define Y_ARITY_MAX 1
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

#ifndef typeof_y0
	#define typeof_y0 dtype
#endif

#if USE_BDA == 0
	#if X_ARITY == 1 && Y_ARITY == 1
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) buffer y0_buf { typeof_y0 y0_data[]; };
	#elif X_ARITY == 2 && Y_ARITY == 1
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
		layout(binding = 2, std430) buffer y0_buf { typeof_y0 y0_data[]; };
	#endif
#endif

layout(push_constant, std430) uniform push
{
	#if USE_BDA
		// x0_data
	#endif
	uint x0_offset;
	uint x0_inc;
	
	#if X_ARITY > 1
		#if USE_BDA
			// x1_data
		#endif
		uint x1_offset;
		uint x1_inc;
	#endif
	#if X_ARITY > 2
		#if USE_BDA
			// x2_data
		#endif
		uint x2_offset;
		uint x2_inc;
	#endif
	
	#if USE_BDA
		// y0_data
	#endif
	uint y0_offset;
	uint y0_inc;
	
	uint total; // total elements
	
	float w[NUM_WEIGHTS];
};

#include "../pointwise-common/pointwise-enum.glsl"

typeof_y0 pointwise_function_unary_unary(typeof_x0 x0)
{
	precise typeof_y0 y0;
	if (POINTWISE_ROUTINE == ROUTINE_IDENTITY)
		y0 = typeof_y0(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_FILL)
		y0 = typeof_y0(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_SCALE)
		y0 = typeof_y0(x0)*typeof_y0(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_ADD_SCALAR)
		y0 = typeof_y0(x0) + typeof_y0(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_SUB_SCALAR)
		y0 = typeof_y0(x0) - typeof_y0(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_DIV_SCALAR)
		y0 = typeof_y0(x0)/typeof_y0(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_RSUB_SCALAR)
		y0 = typeof_y0(w[0]) - typeof_y0(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_RDIV_SCALAR)
		y0 = typeof_y0(w[0])/typeof_y0(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_POW)
		// compute as float, otherwise compiler errors arise too frequently due to missing overloads
		y0 = typeof_y0(pow(dtype(x0), dtype(w[0])));
	else if (POINTWISE_ROUTINE == ROUTINE_AXPB)
		y0 = typeof_y0(w[0]*dtype(x0) + w[1]);
	else if (POINTWISE_ROUTINE == ROUTINE_HARDTANH)
		y0 = typeof_y0(max(typeof_y0(w[0]), min(typeof_y0(w[1]), typeof_y0(x0))));
	else if (POINTWISE_ROUTINE == ROUTINE_ABS)
		y0 = typeof_y0(abs(dtype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_ATAN)
		y0 = typeof_y0(atan(dtype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_LOG)
		y0 = typeof_y0(log(dtype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_SQRT)
		y0 = typeof_y0(sqrt(dtype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_EXP)
		y0 = typeof_y0(exp(dtype(x0)));
	return y0;
}

#if X_ARITY > 1
	typeof_y0 pointwise_function_binary_unary(typeof_x0 x0, typeof_x1 x1)
	{
		// do calculation with dtype, then cast
		precise dtype y0;
		if (POINTWISE_ROUTINE == ROUTINE_ADD)
			y0 = dtype(x0) + dtype(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_SUB)
			y0 = dtype(x0) - dtype(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_MUL)
			y0 = dtype(x0)*dtype(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_DIV)
			y0 = dtype(x0)/dtype(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_AXPY)
			y0 = dtype(x0)*dtype(w[0]) + dtype(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_AXPBY)
			y0 = dtype(dtype(x0)*dtype(w[0]) + dtype(w[1])*dtype(x1));
		else if (POINTWISE_ROUTINE == ROUTINE_HARDTANH_BWD)
			y0 = (dtype(w[0]) <= dtype(x0) && dtype(x0) <= dtype(w[1])) ? dtype(x1) : dtype(0);
		return typeof_y0(y0);
	}
#endif


void pointwise_strided_impl()
{
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= total) return;
	typeof_x0 x0 = x0_data[x0_offset + gid*x0_inc];
	#if X_ARITY > 1
		typeof_x1 x1 = x1_data[x1_offset + gid*x1_inc];
	#endif
	#if X_ARITY > 2
	#endif
	
	uint y0_pos = y0_offset + gid*y0_inc;
	#if X_ARITY == 1
		y0_data[y0_pos] = pointwise_function_unary_unary(x0);
	#elif X_ARITY == 2
		y0_data[y0_pos] = pointwise_function_binary_unary(x0, x1);
	#else
		#error "not implemented"
	#endif
}
