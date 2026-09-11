#include "../pointwise-common/pointwise-enum.glsl"
#ifndef NUM_WEIGHTS_MAX
	#define NUM_WEIGHTS_MAX 8
#endif

#ifndef X_ARITY_MAX
	#define X_ARITY_MAX 3
#endif

#ifndef Y_ARITY_MAX
	#define Y_ARITY_MAX 2
#endif

struct X_IN
{
	acctype data[X_ARITY_MAX];
};

struct Y_OUT
{
	acctype data[Y_ARITY_MAX];
};

struct W_ARGS
{
	float w[NUM_WEIGHTS_MAX];
};

#if USE_SUBGROUP_ARITHMETIC
	// Method for reduce operations on hardware that supports subgroup arithmetic.
	// Input arity is always going to be 1, but methods are technically binary since arguments come from other subgroup invocations
	acctype pointwise_subgroup_reduce(acctype x0, W_ARGS wargs, uint pointwiseRoutine)
	{
		precise acctype y0;
		[[flatten]]
		if (pointwiseRoutine == ROUTINE_ADD)
			y0 = subgroupAdd(x0);
		else if (pointwiseRoutine == ROUTINE_MUL)
			y0 = subgroupMul(x0);
		else if (pointwiseRoutine == ROUTINE_MIN)
			y0 = subgroupMin(x0);
		else if (pointwiseRoutine == ROUTINE_MAX)
			y0 = subgroupMax(x0);
		else if (pointwiseRoutine == ROUTINE_BITWISE_AND)
			// only works for ints :c
			// TODO: figure out how to do better casting for this stuff.
			// Casting from float to int to float again likely results in loss of precision, especially on intel iGPUs
			y0 = acctype(subgroupAnd(iacctype(x0)));
		else if (pointwiseRoutine == ROUTINE_BITWISE_OR)
			y0 = acctype(subgroupOr(iacctype(x0)));
		#if 0
			// whatever this is. probably will not use it.
			else if (pointwiseRoutine == ROUTINE_INCLUSIVE_ADD)
				y0 = subgroupInclusiveAdd(x0);
		#endif
		return y0;
	}
#endif

Y_OUT pointwise_function(Shape pos, uint gid, uint xArity, uint yArity, X_IN xargs, W_ARGS wargs, uint pointwiseRoutine)
{
	precise acctype x0 = xargs.data[0];
	precise acctype x1 = xargs.data[1];
	precise acctype x2 = xargs.data[2];
	precise acctype y0;
	precise acctype y1;
	float w[NUM_WEIGHTS_MAX] = wargs.w;
	
	[[flatten]] // very important
	if (xArity == 1 && yArity == 1)
	{
		[[flatten]]
		if (pointwiseRoutine == ROUTINE_IDENTITY)
			y0 = (x0);
		else if (pointwiseRoutine == ROUTINE_FILL)
			y0 = acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_SCALE)
			y0 = (x0)*acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_ADD_SCALAR)
			y0 = (x0) + acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_SUB_SCALAR)
			y0 = (x0) - acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_DIV_SCALAR)
			y0 = (x0)/acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_RSUB_SCALAR)
			y0 = acctype(w[0]) - acctype(x0);
		else if (pointwiseRoutine == ROUTINE_RDIV_SCALAR)
			y0 = acctype(w[0])/acctype(x0);
		else if (pointwiseRoutine == ROUTINE_POW)
			// compute as float, otherwise compiler errors arise too frequently due to missing overloads
			y0 = acctype(pow(x0, w[0]));
		else if (pointwiseRoutine == ROUTINE_AXPB)
			y0 = acctype(w[0]*acctype(x0) + w[1]);
		else if (pointwiseRoutine == ROUTINE_HARDTANH)
			y0 = acctype(max(acctype(w[0]), min(acctype(w[1]), acctype(x0))));
		else if (pointwiseRoutine == ROUTINE_ABS)
			y0 = acctype(abs(acctype(x0)));
		else if (pointwiseRoutine == ROUTINE_ATAN)
			y0 = acctype(atan(x0));
		else if (pointwiseRoutine == ROUTINE_LOG)
			y0 = acctype(log(acctype(x0)));
		else if (pointwiseRoutine == ROUTINE_SQRT)
			y0 = acctype(sqrt(acctype(x0)));
		else if (pointwiseRoutine == ROUTINE_EXP)
			y0 = acctype(exp(acctype(x0)));
		else if (pointwiseRoutine == ROUTINE_SGN)
			y0 = x0 < A0 ? AN1 : (x0 > A0 ? A1 : A0);
		else if (pointwiseRoutine == ROUTINE_HARDSWISH)
			y0 = x0 <= acctype(-3.0f) ? A0 : (x0 >= acctype(3.0f) ? x0 : x0*(x0+acctype(3.0f))/acctype(6.0f));
		else if (pointwiseRoutine == ROUTINE_HARDSIGMOID)
			y0 = x0 <= acctype(-3.0f) ? A0 : (x0 >= acctype(3.0f) ? A1 : x0/acctype(6.0f) + acctype(0.5f));
		else if (pointwiseRoutine == ROUTINE_SILU)
			y0 = x0 / (A1 + exp(-x0));
		else if (pointwiseRoutine == ROUTINE_LEAKY_RELU)
			y0 = x0 > A0 ? x0 : acctype(w[0]) * x0;
		else if (pointwiseRoutine == ROUTINE_BITWISE_NOT)
			// this one gets weird
			y0 = acctype(~iacctype(x0));
		else if (pointwiseRoutine == ROUTINE_LOGICAL_NOT)
			y0 = x0 > A0 ? A0 : A1;
		else if (pointwiseRoutine == ROUTINE_CLAMP)
			y0 = max(acctype(w[0]), min(acctype(w[1]), x0));
		else if (pointwiseRoutine == ROUTINE_CEIL)
			y0 = ceil(x0);
		else if (pointwiseRoutine == ROUTINE_GELU || pointwiseRoutine == ROUTINE_GELU_APPROXIMATE)
			// For some reason, I simply couldn't get regular GELU to work.
			// Until then, the approximate will always be used.
			y0 = acctype(0.5f) * x0 * (A1 + tanh(acctype(0.7978845608028654f) * x0 * (A1 + acctype(0.044715f) * x0 * x0)));
		else if (pointwiseRoutine == ROUTINE_LOGIT)
		{
			uint use_eps = floatBitsToUint(w[1]);
			if (use_eps > 0)
			{
				acctype z = min(A1 - acctype(w[0]), max(acctype(w[0]), x0));
				y0 = log(z / (A1 - z));
			}
			else
				y0 = log(x0 / (A1 - x0));
		}
		else if (pointwiseRoutine == ROUTINE_ARANGE)
			y0 = acctype(w[0]) + acctype(gid)*acctype(w[1]);
		else if (pointwiseRoutine == ROUTINE_ROUND)
			y0 = round(x0);
		else if (pointwiseRoutine == ROUTINE_NEG)
			y0 = -x0;
		else if (pointwiseRoutine == ROUTINE_RECIP)
			y0 = A1/x0;
		// these get re-used, as to not have too many enum values
		else if (pointwiseRoutine == ROUTINE_CMP_GT)
			y0 = x0 > acctype(w[0]) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_LT)
			y0 = x0 < acctype(w[0]) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_GE)
			y0 = x0 >= acctype(w[0]) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_LE)
			y0 = x0 <= acctype(w[0]) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_EQ)
			y0 = x0 == acctype(w[0]) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_NE)
			y0 = x0 != acctype(w[0]) ? A1 : A0;
		// really need to have a better way of separating integer-only ops...
		else if (pointwiseRoutine == ROUTINE_BITWISE_AND)
			y0 = acctype(iacctype(x0) & iacctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_BITWISE_OR)
			y0 = acctype(iacctype(x0) | iacctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_BITWISE_XOR)
			y0 = acctype(iacctype(x0) ^ iacctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_LOGICAL_AND)
			y0 = acctype(iacctype(x0) & iacctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_LOGICAL_OR)
			y0 = acctype(iacctype(x0) | iacctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_MAX)
			y0 = max(x0, acctype(w[0]));
		else if (pointwiseRoutine == ROUTINE_MIN)
			y0 = min(x0, acctype(w[0]));
	}
	else if (xArity == 2 && yArity == 1)
	{
		[[flatten]]
		if (pointwiseRoutine == ROUTINE_ADD)
			y0 = (x0) + (x1);
		else if (pointwiseRoutine == ROUTINE_SUB)
			y0 = (x0) - (x1);
		else if (pointwiseRoutine == ROUTINE_MUL)
			y0 = (x0)*(x1);
		else if (pointwiseRoutine == ROUTINE_DIV)
			y0 = (x0)/(x1);
		else if (pointwiseRoutine == ROUTINE_AXPY)
			y0 = (x0)*(w[0]) + (x1);
		else if (pointwiseRoutine == ROUTINE_AXPBY)
			y0 = ((x0)*(w[0]) + (w[1])*(x1));
		else if (pointwiseRoutine == ROUTINE_HARDTANH_BWD)
			y0 = ((w[0]) <= (x0) && (x0) <= (w[1])) ? (x1) : acctype(0);
		else if (pointwiseRoutine == ROUTINE_HARDSIGMOID_BWD)
			y0 = (acctype(-3) < x0 && x0 < acctype(3)) ? x1 / acctype(6) : A0;
		else if (pointwiseRoutine == ROUTINE_HARDSWISH_BWD)
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
		else if (pointwiseRoutine == ROUTINE_SILU_BWD)
		{
			y0 = A1 / (A1 + exp(-x0));
			y0 = x1 * y0 * ( A1 + x0 * (A1 - y0));
		}
		else if (pointwiseRoutine == ROUTINE_LEAKY_RELU_BWD)
			y0 = x0 > A0 ? x1 : acctype(w[0]) * x1;
		else if (pointwiseRoutine == ROUTINE_GELU_BWD || pointwiseRoutine == ROUTINE_GELU_APPROXIMATE_BWD)
		{
			// just like the forward, the backward doesn't have the regular GELU implemented :c
			acctype alpha = acctype(1.128379167095512558561f) * acctype(0.7071067811865475f);
			acctype koeff = acctype(0.044715f);
			acctype beta  = alpha * koeff * acctype(3.0f);
			acctype Y = tanh(alpha * fma(koeff, x0*x0*x0,x0));
			y0 = acctype(0.5f) * x1 * fma(fma(-x0, Y*Y, x0), fma(beta, x0*x0, alpha), A1 + Y);
		}
		else if (pointwiseRoutine == ROUTINE_THRESHOLD_BWD)
			y0 = (x0 > acctype(w[0])) ? x1 : A0;
		else if (pointwiseRoutine == ROUTINE_DROPOUT)
			y0 = x0*x1*acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_CMP_GT)
			y0 = (x0 > x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_LT)
			y0 = (x0 < x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_GE)
			y0 = (x0 >= x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_LE)
			y0 = (x0 <= x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_EQ)
			y0 = (x0 == x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_CMP_NE)
			y0 = (x0 != x1) ? A1 : A0;
		else if (pointwiseRoutine == ROUTINE_LERP)
			y0 = x0 + acctype(w[0])*(x1 - x0);
		else if (pointwiseRoutine == ROUTINE_TRANSFORM_BIAS_RESCALE_QKV)
		{
			uint position_d1 = uint(pos.s[1]);
			acctype scale = position_d1 < uint(w[1]) ? acctype(w[0]) : A1;
			y0 = (x0 + x1)*scale;
		}
		else if (pointwiseRoutine == ROUTINE_BITWISE_AND)
			y0 = acctype(iacctype(x0) & iacctype(x1));
		else if (pointwiseRoutine == ROUTINE_BITWISE_OR)
			y0 = acctype(iacctype(x0) | iacctype(x1));
		else if (pointwiseRoutine == ROUTINE_BITWISE_XOR)
			y0 = acctype(iacctype(x0) ^ iacctype(x1));
		else if (pointwiseRoutine == ROUTINE_LOGICAL_AND)
			y0 = acctype(iacctype(x0) & iacctype(x1));
		else if (pointwiseRoutine == ROUTINE_LOGICAL_OR)
			y0 = acctype(iacctype(x0) | iacctype(x1));
		else if (pointwiseRoutine == ROUTINE_MAX)
			y0 = max(x0, x1);
		else if (pointwiseRoutine == ROUTINE_MIN)
			y0 = min(x0, x1);
	}
	else if (xArity == 3 && yArity == 1)
	{
		[[flatten]]
		if (pointwiseRoutine == ROUTINE_LOG_SIGMOID_BWD)
		{
			bool is_negative = x0 < A0;
			acctype maxd = is_negative ? A1: A0;
			acctype s = is_negative ? A1: acctype(-1.0);
			y0 = (maxd - s * (x1 / (A1 + x1))) * x2;
		}
		else if (pointwiseRoutine == ROUTINE_ADDCMUL)
		{
			y0 = fma(acctype(w[0])*x1, x2, x0);
		}
		else if (pointwiseRoutine == ROUTINE_ADDCDIV)
			y0 = x0 + acctype(w[0])*(x1/x2);
		else if (pointwiseRoutine == ROUTINE_MSE_BWD)
			y0 = acctype(2)*(x1 - x2)*x0*acctype(w[0]);
		else if (pointwiseRoutine == ROUTINE_FMA)
			y0 = fma(x0, x1, x2);
		else if (pointwiseRoutine == ROUTINE_BCE_BWD)
			y0 = -(x1 - x0) / max(acctype(1e-12f), x0 - (x0*x0) ) * x2 * acctype(w[0]);
	}
	else if (xArity == 1 && yArity == 2)
	{
		[[flatten]]
		if (pointwiseRoutine == ROUTINE_LOG_SIGMOID)
		{
			y1 = exp(-abs(x0));
			y0 = min(A0, x0) - log(A1 + y1);
		}
	}
	
	
	Y_OUT outp;
	outp.data[0] = y0;
	outp.data[1] = y1;
	return outp;
}
