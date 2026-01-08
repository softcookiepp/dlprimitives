#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "defs.glsl"
#include "workgroup.glsl"

layout(binding = 0, std430) buffer a_buf { dtype a[]; };
layout(binding = 1, std430) buffer c_buf { dtype c[]; };


layout(push_constant, std430) uniform activation
{
	uint64_t size;
#if USE_BDA
	__global dtype *a;
#endif
	uint64_t a_offset;
#if USE_BDA
	__global dtype *c,
#endif
	uint64_t c_offset;
};

void main()
{
    uint64_t pos = get_global_id(0);
    if(pos >= size)
        return;
	uint64_t a_pos = pos + a_offset;
	uint64_t c_pos = pos + c_offset;
	dtype a_val = a[a_pos];
    c[c_pos] = ACTIVATION_F(a_val);
}
