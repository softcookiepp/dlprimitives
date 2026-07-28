#version 450

///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/activation.glsl"

PREPARE_ACTIVATION(3)

#ifndef ELTOP
#define ELTOP 0
#endif

#if ELTOP == 0
#define EOP(x,y) ((x) + (y))
#elif ELTOP == 1
#define EOP(x,y) ((x) * (y))
#elif ELTOP == 2
#define EOP(x,y) max((x),(y))
#else
#error "Invaid operation"
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer a_buf { dtype a[]; };
	layout(binding = 1, std430) readonly buffer b_buf { dtype b[]; };
	layout(binding = 2, std430) writeonly buffer c_buf { dtype c[]; };
#endif

layout(push_constant, std430) uniform eltwise
{
	uint size;
#if USE_BDA
	__global const dtype *a;
#endif
	uint  a_offset;
#if USE_BDA
	__global const dtype *b;
#endif
	uint  b_offset;
#if USE_BDA
	__global dtype *c;
#endif
	uint c_offset;
	dtype c1;
	dtype c2;
};


void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
    dtype value = EOP(a[pos + a_offset]*c1, b[pos + b_offset]*c2);
    c[pos + c_offset] = ACTIVATION_F(value);
}
