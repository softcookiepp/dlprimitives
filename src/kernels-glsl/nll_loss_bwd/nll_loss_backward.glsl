#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

layout(constant_id = 3) const uint REDUCE = 0;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer dx_buf { dtype dx[]; };
	layout(binding = 1, std430) readonly buffer label_buf { itype label[]; };
	layout(binding = 2, std430) readonly buffer dy_buf { dtype dy[]; };
#endif

layout(push_constant, std430) uniform nll_loss_backward
{
	uint batch;
	uint channel;
#if USE_BDA
	dtype_addr_rw dx;
#endif
	uint dx_offset;
#if USE_BDA
	itype_addr_ro label;
#endif
	uint label_offset;
#if USE_BDA
	dtype_addr_ro dy;
#endif
	uint dy_offset;
	dtype scale;
	dtype factor;
};

void main()
{
    uint dx_ = dx_offset;
    uint label_ = label_offset;
    uint dy_ = dy_offset;
    uint c = get_global_id(0);
    uint b = get_global_id(1);
    if(b>= batch || c >= channel)
        return;
    uint index = uint(label[b + label_]);
    uint offset = b*channel + c;
    uint dyoffset = 0;
    if (REDUCE == 0) dyoffset = b;
    dtype dxval = (c == index) ? -scale * dy[dyoffset + dy_] : 0;
    if(factor == 0)
        dx[offset + dx_] = dxval;
    else
        dx[offset + dx_] = dx[offset + dx_]*factor + dxval;
}
