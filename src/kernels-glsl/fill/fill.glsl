#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) writeonly buffer p_buf { dtype p[]; };
#endif

layout(push_constant, std430) uniform fill
{
	uint total;
#if USE_BDA
	__global dtype *p;
#endif
	uint p_offset;
	dtype value;
};

void main()	
{
    uint pos = get_global_id(0);
    if(pos >= total)
        return;
    p[p_offset + pos ] = value;
}
