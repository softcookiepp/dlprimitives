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
layout(binding = 0, std430) readonly buffer bias_buf { dtype bias[]; };

layout(push_constant, std430) uniform ip_bias
{
	int batch;
	int N;
#if USE_BDA
	__global dtype *data;
#endif
	uint data_offset;
#if USE_BDA
	__global const dtype *bias;
#endif
	uint bias_offset;
};

void main()
{
    uint r = get_global_id(0);
    uint c = get_global_id(1);
    if(r >= batch || c >= N)
        return;
	const uint data_pos = (r*N*c) + data_offset;
    const uint bias_pos = c + bias_offset;
    // should this be a float? hmmm...
    float v = data[data_pos] + bias[bias_pos];
    v=ACTIVATION_F(v);
    data[data_pos] = v;
}
