
#ifndef COMMON_GLSL
#define COMMON_GLSL
// =================================================================================================

#define USE_BDA 0
#ifndef dtype
	// Parameters set by the tuner or by the database. Here they are given a basic default value in case
	// this file is used outside of the CLBlast library.
	#ifndef PRECISION
		#define PRECISION 32			// Data-types: half, single or double precision, complex or regular
	#endif

	// =================================================================================================
		
	// reserved for when unrolling semantics are able to be used
	#ifndef UNROLL
		#define UNROLL(N)
	#endif

	// Enable support for half-precision
	#if PRECISION == 16
		#extension GL_EXT_shader_16bit_storage : require
		#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
	#endif

	// Enable support for double-precision
	#if PRECISION == 64 || PRECISION == 6464
		#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
	#endif


	// Half-precision
	#if PRECISION == 16
		struct vec2_t { f16vec2 s; };
		struct vec4_t { f16vec4 s; };
		struct vec8_t { float16_t s[8]; };
		struct vec16_t { float16_t s[16]; };
		#define dtype float16_t
		#define ZERO float16_t(0.0)
		#define ONE float16_t(1.0)
		#define SMALLEST -1.0e14
		#define INFINITY float16_t(uintBitsToFloat(0x7F800000))
		#define NAN float16_t(uintBitsToFloat(0x7FC00000))
		#define PI float16_t(3.14159265358979323846)

	// Single-precision
	#elif PRECISION == 32
		struct vec2_t { vec2 s; };
		struct vec4_t { vec4 s; };
		struct vec8_t { float s[8]; };
		struct vec16_t { float s[16]; };
		#define dtype float
		#define ZERO dtype(0.0f)
		#define ONE 1.0f
		#define SMALLEST -1.0e37f
		#define INFINITY uintBitsToFloat(0x7F800000)
		#define NAN uintBitsToFloat(0x7FC00000)
		#define PI float(3.14159265358979323846)

	// Double-precision 
	#elif PRECISION == 64
		struct vec2_t { dvec2 s; };
		struct vec4_t { dvec4 s; };
		struct vec8_t { double s[8]; };
		struct vec16_t { double s[16]; };
		#define dtype double
		#define ZERO 0.0
		#define ONE 1.0
		#define SMALLEST -1.0e37
		#define INFINITY double(uintBitsToFloat(0x7F800000))
		#define NAN double(uintBitsToFloat(0x7FC00000))
		#define PI double(3.14159265358979323846)

	// Complex single-precision
	#elif PRECISION == 3232
		struct vec2_t { mat2x2 s; };
		struct vec4_t { mat4x2 s; };
		struct vec8_t { vec2 s[8]; };
		struct vec16_t { vec2 s[16]; };
		#define dtype vec2
		#define ZERO 0.0f
		#define ONE 1.0f
		#define SMALLEST -1.0e37f
		#define INFINITY uintBitsToFloat(0x7F800000)
		#define NAN uintBitsToFloat(0x7FC00000)
		#define PI float(3.14159265358979323846)

	// Complex double-precision
	#elif PRECISION == 6464
		struct vec2_t { dmat2x2 s; };
		struct vec4_t { dmat4x2 s; };					 
		struct vec8_t { dvec2 s[8]; };
		struct vec16_t { dvec2 s[16]; };
		#define dtype dvec2
		#define ZERO 0.0
		#define ONE 1.0
		#define SMALLEST -1.0e37
		#define INFINITY double(uintBitsToFloat(0x7F800000))
		#define NAN double(uintBitsToFloat(0x7FC00000))
		#define PI double(3.14159265358979323846)
	#endif

	// this simplifies stuff c:
	#define dtype2 vec2_t
	#define dtype4 vec4_t
	#define dtype8 vec8_t
	#define dtype16 vec16_t

	// Single-element version of a complex number
	#if PRECISION == 3232
		#define singledtype float 
	#elif PRECISION == 6464
		#define singledtype double 
	#else
		#define singledtype dtype 
	#endif

	// Converts a 'dtype argument' value to a 'dtype' value as passed to the kernel. Normally there is no
	// conversion, but half-precision is not supported as kernel argument so it is converted from float.
	#if PRECISION == 16
		#define dtype_arg float 
		#define GetRealArg(x) float16_t(x)
	#else
		#define dtype_arg dtype 
		#define GetRealArg(x) dtype(x)
	#endif

	//#elif PRECISION == 64
	//	// lets see if this makes our life easier...
	//	#define dtype_arg float
	//	#define GetRealArg(x) double(x)

	// Pointers to local memory objects (using a define because CUDA doesn't need them)
	#ifndef LOCAL_PTR
		#define LOCAL_PTR shared
	#endif

	// =================================================================================================

	// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
	// devices, this is enabled (see src/routine.cpp).
	#ifndef USE_CL_MAD
		#define USE_CL_MAD 0
	#endif

	// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
	// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
	#ifndef RELAX_WORKGROUP_SIZE
		#define RELAX_WORKGROUP_SIZE 0
	#endif

	// ensure all spec constants related to workgroup size are here and ready
	#if RELAX_WORKGROUP_SIZE
		layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
	#endif

	// Sets a variable to zero
	#if PRECISION == 3232 || PRECISION == 6464
		#define SetToZero(a) a = dtype(ZERO, ZERO)
	#else
		#define SetToZero(a) a = ZERO
	#endif

	// Sets a variable to zero (only the imaginary part)
	#if PRECISION == 3232 || PRECISION == 6464
		#define ImagToZero(a) a.y = ZERO
	#else
		#define ImagToZero(a) 
	#endif

	// Sets a variable to one
	#if PRECISION == 3232 || PRECISION == 6464
		#define SetToOne(a) a = dtype(ONE, ZERO)
	#else
		#define SetToOne(a) a = ONE
	#endif

	// Determines whether a variable is zero
	#if PRECISION == 3232 || PRECISION == 6464
		#define IsZero(a) ((a[0] == ZERO) && (a[1] == ZERO))
	#else
		#define IsZero(a) (a == ZERO)
	#endif

	// component-wise absolute value
	#define AbsoluteValue(value) value = abs(value)

	// Negation (component-wise)
	#if 0 //PRECISION == 3232 || PRECISION == 6464
		#define Negate(value) value.x = (-1.0(value.x)); value.y = (-1.0*(value.y))
	#else
		#define Negate(value) value = (-1.0*(value))
	#endif

	// Adds two complex variables
	#if 0 //PRECISION == 3232 || PRECISION == 6464
		#define Add(c,a,b) c = dtype(a.x + b.x, a.y + b.y)
	#else
		#define Add(c,a,b) c = a + b
	#endif

	// Subtracts two complex variables
	#if 0 //PRECISION == 3232 || PRECISION == 6464
		#define Subtract(c,a,b) c = dtype(a.x - b.x, a.y - b.y)
	#else
		#define Subtract(c,a,b) c = a - b
	#endif

	// Multiply two complex variables (used in the defines below)
	#if PRECISION == 3232 || PRECISION == 6464
		#define MulReal(a,b) a[0]*b[0] - a[1]*b[1]
		#define MulImag(a,b) a[0]*b[1] + a[1]*b[0]
	#endif

	// The scalar multiply function
	#if PRECISION == 3232 || PRECISION == 6464
		#define Multiply(c,a,b) c = dtype(MulReal(a,b), MulImag(a,b))
	#else
		#define Multiply(c,a,b) c = a * b
	#endif

	#define vMultiply(c, a, b, vWidth) \
	{ \
		UNROLL(vWidth) \
		for (uint i = 0; i < vWidth; i += 1) \
		{ \
			Multiply(c.s[i], a.s[i], b.s[i]); \
		} \
	}

	// c is vector, a is scalar, b is vector
	#define vsMultiply(c, a, b, vWidth) \
	{ \
		UNROLL(vWidth) \
		for (uint i = 0; i < vWidth; i += 1) \
		{ \
			Multiply(c.s[i], a, b.s[i]); \
		} \
	}

	// c is vector, a is scalar, b is vector
	#define vsMultiplyAdd(c, a, b, vWidth) \
	{ \
		UNROLL(vWidth) \
		for (uint i = 0; i < vWidth; i += 1) \
		{ \
			MultiplyAdd(c.s[i], a, b.s[i]); \
		} \
	}

	#define vSetToZero(v, vWidth) \
	{ \
		UNROLL(vWidth) \
		for (uint i = 0; i < vWidth; i += 1) \
		{ \
			SetToZero(v.s[i]); \
		} \
	}

	// The scalar multiply-add function
	#if PRECISION == 3232 || PRECISION == 6464
		#define MultiplyAdd(c,a,b) c += dtype(MulReal(a,b), MulImag(a,b))
	#else
		#if 0 //USE_CL_MAD == 1
			#define MultiplyAdd(c,a,b) c = mad(a, b, c)
		#else
			#define MultiplyAdd(c,a,b) c += (a * b)
		#endif
	#endif

	// The scalar multiply-subtract function
	#if PRECISION == 3232 || PRECISION == 6464
		#define MultiplySubtract(c,a,b) c -= dtype(MulReal(a,b), MulImag(a,b))
	#else
		#define MultiplySubtract(c,a,b) c -= (a * b)
	#endif

	// The scalar division function: full division
	#if PRECISION == 3232 || PRECISION == 6464
		#define DivideFull(c,a,b) singledtype num_x = (a.x * b.x) + (a.y * b.y); singledtype num_y = (a.y * b.x) - (a.x * b.y); singledtype denom = (b.x * b.x) + (b.y * b.y); c = dtype(num_x / denom, num_y / denom)
	#elif PRECISION == 16
		// Some hardware doesn't compute NaN properly
		// #define DivideFull(c,a,b) c = (b == ZERO ? NAN : a / b)
		// still need to test the specifics.
		// until then, just use default behavior...
		#define DivideFull(c,a,b) c = a / b
	#else
		#define DivideFull(c,a,b) c = a / b
	#endif

	// The scalar AXPBY function
	#if PRECISION == 3232 || PRECISION == 6464
		//#define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
		#define AXPBY(e,a,b,c,d) e = dtype(MulReal(a,b) + MulReal(c,d), MulImag(a,b) + MulImag(c,d))
	#else
		#define AXPBY(e,a,b,c,d) e = a*b + c*d
	#endif

	// The complex conjugate operation for complex transforms
	#if PRECISION == 3232 || PRECISION == 6464
		#define COMPLEX_CONJUGATE(value) value.y = -1.0*value.y
	#else
		#define COMPLEX_CONJUGATE(value) 
	#endif
#endif
// =================================================================================================

// Macro for storing and loading, to accomodate BDA
#if USE_BDA
	// this needs to be changed, but I forget how it works
	#define INDEX(buf, idx) buf[idx]
#else
	#define INDEX(buf, idx) buf[idx]
#endif

// =================================================================================================

// vector load methods
#define vload2_single_alignment(index, buf) dtype2(INDEX(buf, index), INDEX(buf, index+1))
#define vload4_signle_alignment(index, buf) dtype4(INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3))

#define vload2(index, buf) dtype2(dtype[2](INDEX(buf, index), INDEX(buf, index+1)))
#define vload4(index, buf) dtype4(dtype[4](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3)))
#define vload8(index, buf) dtype8(dtype[8](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7)))
#define vload16(index, buf) dtype16(dtype[16](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7),\
	INDEX(buf, index+8), INDEX(buf, index+9), INDEX(buf, index+10), INDEX(buf, index+11), INDEX(buf, index+12),\
	INDEX(buf, index+13), INDEX(buf, index+14), INDEX(buf, index+15) ))
	
#define vloadN(index, buf, N) vload##N(index, buf)

#define vTranspose(dst, src, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		UNROLL(vWidth) \
		for (uint j = 0; j < vWidth; j += 1) dst[i].s[j] = src[j].s[i]; \
	} \
}

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
	#define USE_STAGGERED_INDICES 0
#endif

#ifndef CL_ALIASES
#define CL_ALIASES
#define get_global_id(dim) gl_GlobalInvocationID[dim]
#define get_local_id(dim) gl_LocalInvocationID[dim]
#define get_group_id(dim) gl_WorkGroupID[dim]
#define get_global_size(idx) gl_NumWorkGroups[idx] * gl_WorkGroupSize[idx]
#define get_local_size(idx) gl_WorkGroupSize[idx]
#define get_num_groups(dim) gl_NumWorkGroups[dim]
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
	uint GetGroupIDFlat() {
		return get_group_id(0) + get_num_groups(0) * get_group_id(1);
		//return gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y;
	}
	uint GetGroupID1() {
		return (GetGroupIDFlat()) % get_num_groups(1);
		//return int((GetGroupIDFlat()) % gl_NumWorkGroups.y);
	}
	uint GetGroupID0() {
		return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
		//return int(((GetGroupIDFlat() / gl_NumWorkGroups.y) + GetGroupID1()) % gl_WorkGroupSize.x);
	}
#else
	uint GetGroupID1() { return get_group_id(1); }
	//int GetGroupID1() { return int(gl_WorkGroupID.y); }
	uint GetGroupID0() { return get_group_id(0); }
	//int GetGroupID0() { return int(gl_WorkGroupID.x); }
#endif

// stubs to keep the compiler happy; they in fact do nothing, however
int64_t exp(int64_t x) { return int64_t(0); }
uint64_t exp(uint64_t x) { return uint64_t(0); }
int exp(int x) { return int(0); }
uint exp(uint x) { return uint(0); }

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
	x = abs(x);
	dtype t = dtype( dtype(1.0)/(dtype(1.0) + p*x) );
	precise dtype y = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp( (-x)*x);
	return sign*y;
}


// End of the C++11 raw string literal
#endif
//)"

// =================================================================================================
