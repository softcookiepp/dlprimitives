#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/activation.glsl"

PREPARE_ACTIVATION(3)

layout(binding = 0, std430) buffer a_buf { dtype a[]; };
layout(binding = 1, std430) buffer c_buf { dtype c[]; };


layout(push_constant, std430) uniform activation
{
	uint size;
#if USE_BDA
	dtype_addr_rw a;
#endif
	uint a_offset;
#if USE_BDA
	dtype_addr_rw c;
#endif
	uint c_offset;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
	uint a_pos = pos + a_offset;
	uint c_pos = pos + c_offset;
	dtype a_val = a[a_pos];
    c[c_pos] = ACTIVATION_F(a_val);
}
