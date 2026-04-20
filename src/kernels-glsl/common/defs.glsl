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

#define ACTIVATION_IDENTITY 0
#define ACTIVATION_RELU     1
#define ACTIVATION_TANH     2
#define ACTIVATION_SIGMOID  3
#define ACTIVATION_RELU6    4

#ifndef dtype
	#define sizeof_dtype 4
	#define dtype float
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
#endif

#define cmp_gt(a, b) (dtype(a) > dtype(b) )

#ifndef itype
	// just default to 32 bit for now just so we can test to ensure it compiles
	#define itype int
#endif


#ifndef ACTIVATION
#define ACTIVATION ACTIVATION_IDENTITY
#endif

#if ACTIVATION == ACTIVATION_IDENTITY
#   define ACTIVATION_F(x) (x)
#   define ACTIVATION_FINV(y,dy) (dy)
#   define ACTIVATION_NAME identity
#elif ACTIVATION == ACTIVATION_RELU
#   define ACTIVATION_F(x) (max((x),dtype(0)))
#   define ACTIVATION_FINV(y,dy)  ((y>0)?dy:0)
#   define ACTIVATION_NAME relu
#elif ACTIVATION == ACTIVATION_TANH
#   define ACTIVATION_F(x) (tanh((x)))
#   define ACTIVATION_FINV(y,dy) ((1-(y)*(y))*(dy))
#   define ACTIVATION_NAME tanh 
#elif ACTIVATION == ACTIVATION_SIGMOID
#   define ACTIVATION_F(x) (dtype(1) / (dtype(1) + exp(-(x))))
#   define ACTIVATION_FINV(y,dy) ((y)*(1-(y))*(dy))
#   define ACTIVATION_NAME sigmoid
#elif ACTIVATION == ACTIVATION_RELU6
#   define ACTIVATION_F(x) (min(max((x),dtype(0)), dtype(6)))
#   define ACTIVATION_FINV(y,dy)  ((0<y && y<6)?dy:0)
#   define ACTIVATION_NAME relu6
#else
#   error "Unknown activation"
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
