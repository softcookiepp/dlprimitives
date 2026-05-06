#version 450
#include "../common/defs.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer c_buf {dtype c[]; };
	layout(binding = 1, std430) readonly buffer bias_buf { dtype bias[]; };
#endif

layout(push_constant, std430) uniform bias_fwd
{
#if USE_BDA
	// c
#endif
	uint c_offset;
#if USE_BDA
	// bias
#endif
	uint bias_offset;
	
};

void main()
{
	// workgroup size should be N by M
	uint n_pos = get_workgroup_id(0);
	uint m_pos = get_workgroup_id(1);
	
	// load C
	dtype cvalue = c[c_offset + ];
}
