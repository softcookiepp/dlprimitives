#if !defined(GGML_TYPES_COMP)
#define GGML_TYPES_COMP

// TODO: see if these can be moved
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require

#if PRECISION == 32
	#define QUANT_K 1
	#define QUANT_R 1

	#if LOAD_VEC_A == 4
		#define A_TYPE vec4
	#elif LOAD_VEC_A == 8
		#define A_TYPE mat2x4
	#else
		#define A_TYPE float
	#endif
#elif PRECISION == 16
	#define QUANT_K 1
	#define QUANT_R 1

	#if LOAD_VEC_A == 4
		#define A_TYPE f16vec4
	#elif LOAD_VEC_A == 8
		#define A_TYPE f16mat2x4
	#else
		#define A_TYPE float16_t
	#endif
#elif dtype == bfloat16_t
	#define QUANT_K 1
	#define QUANT_R 1

	#if LOAD_VEC_A == 4
	#define A_TYPE u16vec4
	#elif LOAD_VEC_A == 8
	#error unsupported
	#else
	#define A_TYPE uint16_t
	#endif
#endif


#define B_TYPE A_TYPE
#define D_TYPE A_TYPE

// returns the bfloat value in the low 16b.
// See ggml_compute_fp32_to_bf16
uint32_t fp32_to_bf16(float f)
{
    uint32_t u = floatBitsToUint(f);
    u = (u + (0x7fff + ((u >> 16) & 1))) >> 16;
    return u;
}

float bf16_to_fp32(uint32_t u)
{
    return uintBitsToFloat(u << 16);
}

vec4 bf16_to_fp32(uvec4 u)
{
    return vec4(bf16_to_fp32(u.x), bf16_to_fp32(u.y), bf16_to_fp32(u.z), bf16_to_fp32(u.w));
}

float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = x;
        bits = bits << 23;
    }

    return uintBitsToFloat(bits);
}

#if USE_BDA

#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#define BDA_STORAGE_T uint64_t
#define BDA_OFFSET_T uint64_t

#else

#define BDA_STORAGE_T uvec2
#define BDA_OFFSET_T uint

#endif

#endif // !defined(GGML_TYPES_COMP)
