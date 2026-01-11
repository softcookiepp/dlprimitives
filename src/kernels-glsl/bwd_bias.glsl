#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "common/defs.glsl"
#include "common/reduce.glsl"

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#ifndef SIZE_2D
#define SIZE_2D 1
#endif


layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer dy_buf { dtype dy[]; };
	layout(binding = 1, std430) buffer dx_buf { dtype dx[]; };
#endif

layout(push_constant, std430) uniform bwd_bias
{
	uint features; uint over;
#if USE_BDA
	__global dtype *dy;
#endif
	uint dy_offset;
#if USE_BDA
	__global dtype *dx;
#endif
	uint dx_offset; int dx_stride; float beta;
};

REDUCE_PREPARE(WG_SIZE,dtype);

void main()
{
    uint feature = get_global_id(1);
    if(feature >= features)
        return;

    uint dx_pos = get_group_id(0);
    if(dx_pos >= dx_stride)
        return;

    uint position   = get_global_id(0) * ITEMS_PER_WI;

    dtype val = 0;
    uint batch_scale = features * SIZE_2D;
    #pragma unroll
    for(uint i=0;i<ITEMS_PER_WI;i++) {
        uint index = position + i;
        if(index >= over)
            continue;

        uint batch = index / SIZE_2D;
        uint rcpos = index % SIZE_2D;
        val += dy[batch * batch_scale + feature * SIZE_2D + rcpos + dy_offset];
    }
    
    my_work_group_reduce_add(val);

    uint pos = feature * dx_stride + dx_pos;
    
    if(get_local_id(0) == 0) {
        if(beta == 0)
            dx[pos + dx_offset] = val;
        else
            dx[pos + dx_offset] = dx[feature + dx_offset] * beta + val;
    }
}

