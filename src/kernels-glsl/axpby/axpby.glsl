#version 450
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer x_buf { dtype x[]; };
	layout(binding = 1, std430) buffer y_buf { dtype y[]; };
	layout(binding = 2, std430) buffer z_buf { dtype z[]; };
#endif

layout(push_constant, std430) uniform axpby
{
	uint size;
	dtype a;
#if USE_BDA
	dtype_addr_ro x;
#endif
	uint x_off;
	dtype b;
#if USE_BDA
	dtype_addr_ro y;
#endif
	uint y_off;
#if USE_BDA
	dtype_addr_rw z;
#endif
	uint z_off;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
    z[pos + z_off] = a*x[pos + x_off] + b*y[pos + y_off];
}
