#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "common/defs.glsl"
#include "common/workgroup.glsl"

layout(binding = 0, std430) buffer x_buf { dtype x[]; };
layout(binding = 1, std430) buffer y_buf { dtype y[]; };
layout(binding = 2, std430) buffer z_buf { dtype z[]; };

layout(push_constant, std430) uniform axpby
{
	uint size;
	dtype a;
#if USE_BDA
	__global const dtype *x;
#endif
	uint x_off;
	dtype b;
#if USE_BDA
	__global const dtype *y;
#endif
	uint y_off;
#if USE_BDA
	__global dtype *z;
#endif
	uint z_off;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
    z[pos + z_off] = a*x[pos + x_off] + b*y[pos + y_off];
}
