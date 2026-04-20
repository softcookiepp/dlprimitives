#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/common.glsl"

#if USE_BDA == 0
BUFFER_DEFS
#endif

// so this is going to be extremely difficult :c
layout(push_constant, std430) uniform exec
{
	uint total;
	PARAMS
};

void main()
{
    uint index=get_global_id(0);
    if(index>=total)
        return;
    LOADS
    CALC
    SAVES
}

