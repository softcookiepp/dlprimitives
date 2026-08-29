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

#define Y_ARITY_MAX 2
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
#ifndef typeof_y1
	#define typeof_y1 dtype
#endif

#if USE_BDA == 0
	#if X_ARITY == 1
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) buffer y0_buf { typeof_y0 y0_data[]; };
		#if Y_ARITY > 1
			layout(binding = 2, std430) buffer y1_buf { typeof_y1 y1_data[]; };
		#endif
	#elif X_ARITY == 2
		layout(binding = 0, std430) readonly buffer x0_buf { typeof_x0 x0_data[]; };
		layout(binding = 1, std430) readonly buffer x1_buf { typeof_x1 x1_data[]; };
		layout(binding = 2, std430) buffer y0_buf { typeof_y0 y0_data[]; };
		#if Y_ARITY > 1
			layout(binding = 3, std430) buffer y1_buf { typeof_y1 y1_data[]; };
		#endif
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
	#if Y_ARITY > 1
		#if USE_BDA
			// y1_data
		#endif
		uint y1_offset;
		uint y1_inc;
	#endif
	
	uint total; // total elements
	
	float w[NUM_WEIGHTS];
};

#include "../pointwise-common/pointwise-enum.glsl"

struct Y_OUT
{
	acctype data[Y_ARITY];
};

acctype pointwise_function_unary_unary(precise acctype x0)
{
	precise acctype y0;
	if (POINTWISE_ROUTINE == ROUTINE_IDENTITY)
		y0 = (x0);
	else if (POINTWISE_ROUTINE == ROUTINE_FILL)
		y0 = acctype(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_SCALE)
		y0 = (x0)*acctype(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_ADD_SCALAR)
		y0 = (x0) + acctype(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_SUB_SCALAR)
		y0 = (x0) - acctype(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_DIV_SCALAR)
		y0 = (x0)/acctype(w[0]);
	else if (POINTWISE_ROUTINE == ROUTINE_RSUB_SCALAR)
		y0 = acctype(w[0]) - acctype(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_RDIV_SCALAR)
		y0 = acctype(w[0])/acctype(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_POW)
		// compute as float, otherwise compiler errors arise too frequently due to missing overloads
		y0 = acctype(pow(x0, w[0]));
	else if (POINTWISE_ROUTINE == ROUTINE_AXPB)
		y0 = acctype(w[0]*acctype(x0) + w[1]);
	else if (POINTWISE_ROUTINE == ROUTINE_HARDTANH)
		y0 = acctype(max(acctype(w[0]), min(acctype(w[1]), acctype(x0))));
	else if (POINTWISE_ROUTINE == ROUTINE_ABS)
		y0 = acctype(abs(acctype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_ATAN)
		y0 = acctype(atan(x0));
	else if (POINTWISE_ROUTINE == ROUTINE_LOG)
		y0 = acctype(log(acctype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_SQRT)
		y0 = acctype(sqrt(acctype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_EXP)
		y0 = acctype(exp(acctype(x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_SGN)
		y0 = x0 < A0 ? AN1 : (x0 > A0 ? A1 : A0);
	else if (POINTWISE_ROUTINE == ROUTINE_HARDSWISH)
		y0 = x0 <= acctype(-3.0f) ? A0 : (x0 >= acctype(3.0f) ? x0 : x0*(x0+acctype(3.0f))/acctype(6.0f));
	else if (POINTWISE_ROUTINE == ROUTINE_HARDSIGMOID)
		y0 = x0 <= acctype(-3.0f) ? A0 : (x0 >= acctype(3.0f) ? A1 : x0/acctype(6.0f) + acctype(0.5f));
	else if (POINTWISE_ROUTINE == ROUTINE_SILU)
		y0 = x0 / (A1 + exp(-x0));
	else if (POINTWISE_ROUTINE == ROUTINE_LEAKY_RELU)
		y0 = x0 > A0 ? x0 : acctype(w[0]) * x0;
	else if (POINTWISE_ROUTINE == ROUTINE_BITWISE_NOT)
		// this one gets weird
		y0 = acctype(~iacctype(x0));
	else if (POINTWISE_ROUTINE == ROUTINE_LOGICAL_NOT)
		y0 = x0 > A0 ? A0 : A1;
	else if (POINTWISE_ROUTINE == ROUTINE_CLAMP)
		y0 = max(acctype(w[0]), min(acctype(w[1]), x0));
	else if (POINTWISE_ROUTINE == ROUTINE_CEIL)
		y0 = ceil(x0);
	else if (POINTWISE_ROUTINE == ROUTINE_GELU || POINTWISE_ROUTINE == ROUTINE_GELU_APPROXIMATE)
		// For some reason, I simply couldn't get regular GELU to work.
		// Until then, the approximate will always be used.
		y0 = acctype(0.5f) * x0 * (A1 + tanh(acctype(0.7978845608028654f) * x0 * (A1 + acctype(0.044715f) * x0 * x0)));
	else if (POINTWISE_ROUTINE == ROUTINE_LOGIT)
	{
		precise acctype eps = acctype(w[0]);
		precise acctype use_eps = acctype(w[1]);
		precise acctype z;
		if (use_eps > A0)
			z = min(A1 - eps, max(eps, x0));
		else
			z = x0;
		y0 = log(z / (A1 - z)); 
	}
	return y0;
}

#if X_ARITY > 1
	typeof_y0 pointwise_function_binary_unary(acctype x0, acctype x1)
	{
		precise acctype y0;
		if (POINTWISE_ROUTINE == ROUTINE_ADD)
			y0 = (x0) + (x1);
		else if (POINTWISE_ROUTINE == ROUTINE_SUB)
			y0 = (x0) - (x1);
		else if (POINTWISE_ROUTINE == ROUTINE_MUL)
			y0 = (x0)*(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_DIV)
			y0 = (x0)/(x1);
		else if (POINTWISE_ROUTINE == ROUTINE_AXPY)
			y0 = (x0)*(w[0]) + (x1);
		else if (POINTWISE_ROUTINE == ROUTINE_AXPBY)
			y0 = ((x0)*(w[0]) + (w[1])*(x1));
		else if (POINTWISE_ROUTINE == ROUTINE_HARDTANH_BWD)
			y0 = ((w[0]) <= (x0) && (x0) <= (w[1])) ? (x1) : acctype(0);
		else if (POINTWISE_ROUTINE == ROUTINE_HARDSIGMOID_BWD)
			y0 = (acctype(-3) < x0 && x0 < acctype(3)) ? x1 / acctype(6) : A0;
		else if (POINTWISE_ROUTINE == ROUTINE_HARDSWISH_BWD)
		{
			if (x0 < acctype(-3.0f))
			{
				y0 = A0;
			}
			else if (x0 <= acctype(3.0f))
			{
				y0 =  x1 * ((x0 / acctype(3.0f)) + acctype(0.5f));
			}
			else
			{
				y0 = x1;
			}
		}
		else if (POINTWISE_ROUTINE == ROUTINE_SILU_BWD)
		{
			y0 = A1 / (A1 + exp(-x0));
			y0 = x1 * y0 * ( A1 + x0 * (A1 - y0));
		}
		else if (POINTWISE_ROUTINE == ROUTINE_LEAKY_RELU_BWD)
			y0 = x0 > A0 ? x1 : acctype(w[0]) * x1;
		else if (POINTWISE_ROUTINE == ROUTINE_GELU_BWD || POINTWISE_ROUTINE == ROUTINE_GELU_APPROXIMATE_BWD)
		{
			// just like the forward, the backward doesn't have the regular GELU implemented :c
			acctype alpha = acctype(1.128379167095512558561f) * acctype(0.7071067811865475f);
			acctype koeff = acctype(0.044715f);
			acctype beta  = alpha * koeff * acctype(3.0f);
			acctype Y = tanh(alpha * fma(koeff, x0*x0*x0,x0));
			y0 = acctype(0.5f) * x1 * fma(fma(-x0, Y*Y, x0), fma(beta, x0*x0, alpha), A1 + Y);
		}
		return typeof_y0(y0);
	}
#endif

#if Y_ARITY > 1
	Y_OUT pointwise_function_unary_binary(acctype x0)
	{
		Y_OUT y;
		if (POINTWISE_ROUTINE == ROUTINE_LOG_SIGMOID)
		{
			y.data[1] = exp(-abs(x0));
			y.data[0] = min(A0, x0) - log(A1 + y.data[1]);
		}
		return y;
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
	#if Y_ARITY == 1
		#if X_ARITY == 1
			y0_data[y0_pos] = typeof_y0(pointwise_function_unary_unary(acctype(x0)));
		#elif X_ARITY == 2
			y0_data[y0_pos] = pointwise_function_binary_unary(acctype(x0), acctype(x1));
		#else
			#error "not implemented"
		#endif
	#elif Y_ARITY == 2
		uint y1_pos = y1_offset + gid*y1_inc;
		#if X_ARITY == 1
			Y_OUT y_out = pointwise_function_unary_binary(acctype(x0));
			y0_data[y0_pos] = typeof_y0(y_out.data[0]);
			y1_data[y1_pos] = typeof_y1(y_out.data[1]);
		#else
			#error "not implemented"
		#endif
	#endif
}
