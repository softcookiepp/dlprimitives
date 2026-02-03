#version 450
#include "../common/defs.glsl"
#include "../common/broadcast_dims.glsl"

#if USE_BDA == 0
BUFFER_DEFS
#endif

uint get_offset(Shape s,Shape strides,uint offset)
{
    uint r = offset;
    UNROLL(DIMS)
    for(uint i=0;i<DIMS;i++) {
        r+= s.s[i]*strides.s[i];
    }
    return r;
}

uint get_direct_offset(Shape s,Shape sizes,uint offset)
{
    uint index = 0;
    UNROLL(DIMS)
    for(uint i=0;i<DIMS-1;i++) {
        index += s.s[i];
        index *= sizes.s[i+1];
    }
    index += s.s[DIMS-1] + offset;
    return index;
}

#define get_pos(limits) get_pos_broadcast(limits)

bool valid_pos(Shape pos,Shape limits)
{
    UNROLL(DIMS)
    for(uint i=0;i<DIMS;i++)
        if(pos.s[i] >= limits.s[i])
            return false;
    return true;

}

layout(push_constant, std430) uniform exec
{
	Shape limit;
	PARAMS
};

void main()
{
    Shape index = get_pos(limit);
    if(!valid_pos(index,limit)) {
        return;
    }
    LOADS
    CALC
    SAVES
}


