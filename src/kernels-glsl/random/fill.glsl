#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#define RANDOM_TYPE_UNIFORM 0
#define RANDOM_TYPE_NORMAL 1
#define RANDOM_TYPE_BERNOULLI 2

layout(constant_id = 3) const uint RANDOM_TYPE = 1;


uint mulhi(uint a,uint b)
{
    uint64_t v = uint64_t(a);
    v*=uint64_t(b);
    return uint(v>>32);
}

uint mullo(uint a,uint b)
{
    uint64_t v = uint64_t(a);
    v*=uint64_t(b);
    return uint(v);
}


struct state {
    uint l0,r0,l1,r1;
    uint k0,k1;
};

state single_round(state s)
{
    state next;
    next.l1 = mullo(s.r1, 0xD2511F53);
    next.r1 = mulhi(s.r0, 0xCD9E8D57) ^ s.k0 ^ s.l0;
    next.l0 = mullo(s.r0, 0xCD9E8D57);
    next.r0 = mullo(s.r1, 0xD2511F53) ^ s.k1 ^ s.l1;
    next.k0 = s.k0 + 0xBB67AE85;
    next.k1 = s.k1 + 0x9E3779B9;
    return next;
}

state make_initial_state(uint64_t seed,uint64_t sequence)
{
    state s;
    s.l1 = uint(sequence >> 32);
    s.r1 = uint(sequence);
    s.l0 = 0;
    s.r0 = 0;
    s.k0 = uint(seed);
    s.k1 = uint(seed >> 32);
    return s;
}

uvec4 calculate(state s)
{
    UNROLL(10)
    for(int i=0;i<10;i++)
        s=single_round(s);
    uvec4 r;
    r[0] = s.l0;
    r[1] = s.r0;
    r[2] = s.l1;
    r[3] = s.r1;
    return r;
}

vec4 calculate_float(state s)
{
    uvec4 r = calculate(s);
    vec4 f;
    /// make sure float does not become 1 after rounding
    /// 24 - for float/bfloat16
    /// 16 - for half
    /// 32 - for double
    uint64_t accuracy_shift = 24;
    uint drop_bits = uint(32 - accuracy_shift);
    float factor = 1.0f / float(uint64_t(1) << accuracy_shift);
    f[0] = (r[0] >> drop_bits) * factor;
    f[1] = (r[1] >> drop_bits) * factor;
    f[2] = (r[2] >> drop_bits) * factor;
    f[3] = (r[3] >> drop_bits) * factor;
    return f;
}

vec2 normal_pair(vec2 v)
{
    float scale = sqrt(-2.0f*log(1.0f - v[0]));
    float angle = (2.0f*3.1415926535f)*v[1];
    return vec2(scale*cos(angle), scale*sin(angle));
}

#if USE_BDA == 0
	layout(binding = 0, std430) writeonly buffer p_buf { float p[]; };
#endif

layout(push_constant, std430) uniform fill
{
	uint64_t total;
#if USE_BDA
	__global float *p;
#endif
	uint p_offset;
	uint64_t seed;
	uint64_t init_seq;
	float v1;
	float v2;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos * 4 >= total)
        return;
    uint p_ = p_offset;
    uint64_t seq = init_seq + uint64_t(pos);
    state s = make_initial_state(seed, seq);
    vec4 r = calculate_float(s);

	// specialization constants will eliminate flow control paths entirely
	if (RANDOM_TYPE == RANDOM_TYPE_UNIFORM)
	{
		r = r * vec4(v2-v1) + vec4(v1);
	}
	else if(RANDOM_TYPE == RANDOM_TYPE_NORMAL)
	{
		r.xy = normal_pair(r.xy);
		r.zw = normal_pair(r.zw);
		r = r*vec4(v2) + vec4(v1);
	}
	else if(RANDOM_TYPE == RANDOM_TYPE_BERNOULLI)
	{
		r[0] = r[0] < v1 ? 1:0;
		r[1] = r[1] < v1 ? 1:0;
		r[2] = r[2] < v1 ? 1:0;
		r[3] = r[3] < v1 ? 1:0;
	}
    uint index = pos * 4;
#if USE_BDA // there is no vstore in GLSL, afaik. However, if BDA is used, it will be possible to pointer cast
    if(index < total) {
        vstore4(r,0,p + index);
    }
    else
#endif
    {
        if(index + 0 < total) p[index + 0 + p_]=r[0];
        if(index + 1 < total) p[index + 1 + p_]=r[1];
        if(index + 2 < total) p[index + 2 + p_]=r[2];
        if(index + 3 < total) p[index + 3 + p_]=r[3];
    }
}
