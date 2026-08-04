#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer p_buf { dtype p[]; };
#endif

layout(push_constant, std430) uniform sscal
{
	uint size;
	dtype scale;
#if USE_BDA
	__global dtype *p;
#endif
	uint p_off;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
	p[pos + p_off] = p[pos + p_off] * scale;
}
