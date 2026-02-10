#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer data_buf { dtype data[]; };
#endif

layout(push_constant, std430) uniform activation_inplace
{
	uint tensor_size;
#if USE_BDA
	dtype_addr_rw data;
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

       
