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

layout(binding = 0, std430) buffer data_buf { dtype data[]; };

layout(push_constant, std430) uniform activation_inplace
{
	int tensor_size;
#if USE_BDA
	__global dtype *data;
#endif
	uint data_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= tensor_size)
        return;
    dtype dval = data[pos + data_offset];
    data[pos + data_offset] = ACTIVATION_F(dval);
}

       
