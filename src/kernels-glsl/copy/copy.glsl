#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer target_buf { dtype target[]; };
	layout(binding = 1, std430) buffer source_buf { dtype source[]; };
#endif

layout(push_constant, std430) uniform copy
{
	uint slice;
	uint dim0;
	uint dim1_tgt;uint dim1_tgt_offset;
	uint dim1_src;uint dim1_src_offset;
	uint dim2;
#if USE_BDA
	dtype_addr_rw target;
#endif
	uint target_offset;
#if USE_BDA
	dtype_addr_ro source;
#endif
	uint source_offset;
	dtype scale;
};
void main()
{
    uint p0 = get_global_id(0);
    uint p1 = get_global_id(1);
    uint p2 = get_global_id(2);
    if(p0 >= dim2 || p1 >= slice || p2 >= dim0)
        return;
	uint _target_offset = target_offset + (p0 + (p1 + dim1_tgt_offset) * dim2 + p2 * (dim1_tgt * dim2));
    uint _source_offset = source_offset + (p0 + (p1 + dim1_src_offset) * dim2 + p2 * (dim1_src * dim2));
    if(scale == 0.0)
        target[_target_offset] = source[_source_offset];
    else
        target[_target_offset] = scale * target[_target_offset] + source[_source_offset];
}
