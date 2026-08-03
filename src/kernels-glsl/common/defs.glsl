#ifndef COMMON_GLSL
#define COMMON_GLSL
	// just to prevent the other header from interfering
#endif

// constant thingies
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308

// whether or not to use float32 atomics.
#ifndef ATOMIC_FLOAT32
	#define ATOMIC_FLOAT32 0
#endif
#if ATOMIC_FLOAT32
	#extension GL_EXT_shader_atomic_float : require
#endif

// this is not implemented yet, if it ever will be
#ifndef USE_BDA
	#define USE_BDA 0
#endif

// OpenCL semantics, because why not
#ifndef CL_ALIASES
	#define CL_ALIASES
	#define get_global_id(dim) gl_GlobalInvocationID[dim]
	#define get_local_id(dim) gl_LocalInvocationID[dim]
	#define get_group_id(dim) gl_WorkGroupID[dim]
	#define get_global_size(idx) gl_NumWorkGroups[idx] * gl_WorkGroupSize[idx]
	#define get_local_size(idx) gl_WorkGroupSize[idx]
	#define get_num_groups(dim) gl_NumWorkGroups[dim]
#endif

// we are porting cuda kernels from torch too, because some of the OpenCL kernels from dlprimitives are just too stupid to be portable.
// so we are gonna need this, at least temporarily
#ifndef CUDA_ALIASES
	#define CUDA_ALIASES
	#define threadIdx gl_LocalInvocationID
	#define blockIdx gl_WorkGroupID
	#define blockDim gl_WorkGroupSize
	#define gridDim gl_NumWorkGroups
#endif

#ifndef dtype
	#define dtype float
#endif

#if dtype == float
	#define sizeof_dtype 4
	#define PRECISION 32
	#define dtype2 vec2
	#define dtype4 vec4
	#define DTYPE_MAX FLT_MAX
	#define DTYPE_MIN FLT_MIN
	#if ATOMIC_FLOAT32
		#define atomic_dtype dtype
		#define dtype_to_atomic(v) v
		#define atomic_to_dtype(v) v
	#else
		#define atomic_dtype uint
		#define dtype_to_atomic(v) floatBitsToUint(v)
		#define atomic_to_dtype(v) uintBitsToFloat(v)
	#endif
	
	#define INFINITY uintBitsToFloat(0x7F800000)
	#define NAN uintBitsToFloat(0x7FC00000)
#else
	#error "dtype constants not implemented"
#endif

#define cmp_gt(a, b) (dtype(a) > dtype(b) )

#ifndef itype
	// just default to 32 bit for now just so we can test to ensure it compiles
	#define itype int
#endif


#ifndef USE_UNROLL
	#define USE_UNROLL 0
#endif

#ifndef UNROLL
	#if USE_UNROLL
		#define UNROLL(d)
	#else
		#define UNROLL(d)
	#endif
#endif

// stubs to keep the compiler happy; they in fact do nothing, however
int64_t exp(int64_t x) { return int64_t(0); }
uint64_t exp(uint64_t x) { return uint64_t(0); }
int exp(int x) { return int(0); }
uint exp(uint x) { return uint(0); }

double exp(double x) { precise float y = float(x); y = exp(y); return double(y); }

dtype erf(dtype x)
{
	// adapted from: https://www.johndcook.com/blog/python_erf/
	dtype a1 =  dtype(0.254829592);
	dtype a2 = dtype(-0.284496736);
	dtype a3 =  dtype(1.421413741);
	dtype a4 = dtype(-1.453152027);
	dtype a5 =  dtype(1.061405429);
	dtype p  =  dtype(0.3275911);
	
	precise dtype sign = x > dtype(0.0) ? dtype(1.0) : dtype(-1.0);
	x = dtype(abs(x));
	dtype t = dtype( dtype(1.0)/(dtype(1.0) + p*x) );
	precise dtype y = dtype( (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp( (-x)*x) );
	return sign*y;
}

// borrowed from pytorch
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type) \
	uint _i_n_d_e_x = (gl_WorkGroupID.x * gl_WorkGroupSize.x) + gl_LocalInvocationID.x; \
	for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+= gl_WorkGroupSize.x * gl_NumWorkGroups.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, uint)
