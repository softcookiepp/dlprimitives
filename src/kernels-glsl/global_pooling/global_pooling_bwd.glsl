#version 450
#include "../common/defs.glsl"
#include "../common/reduce.glsl"

layout(local_size_x = 1, local_size_y_id = 0, local_size_z = 1) in;
layout(constant_id = 0) const uint WG_SIZE = 256;
layout(constant_id = 1) const uint ITEMS_PER_WI = 1;
layout(constant_id = 2) const uint POOL_MODE = 1;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x_buf { dtype x[]; };
	layout(binding = 1, std430) buffer dx_buf { dtype dx[]; };
	layout(binding = 2, std430) readonly buffer outp_buf { dtype outp[]; };
#endif

layout(push_constant, std430) uniform global_pooling_bwd
{
	uint items; uint over; float scale;
#if USE_BDA
	__global const dtype *x; // readonly
#endif
	uint  x_offset;
#if USE_BDA
	__global dtype *dx; // read + write
#endif
	uint  dx_offset;
#if USE_BDA
	__global const dtype *outp; // readonly
#endif
	uint  out_offset;
	dtype factor;
};

REDUCE_PREPARE(WG_SIZE, dtype);
shared uint reduce_indx[WG_SIZE];
shared dtype reduce_vals[WG_SIZE];
        
void main()
{
    uint x_ = x_offset;
    uint dx_ = dx_offset;
    uint out_ = out_offset;
    
    uint b = get_global_id(0);

    if(b >= items)
        return;

    uint c = get_global_id(1) * ITEMS_PER_WI;

    if (POOL_MODE == 0)
	{
		x_  += b * over;
    }
    dx_ += b * over;
    out_ += b;

	if (POOL_MODE == 0)
	{
		dtype val = -DTYPE_MAX;
		uint index = -1;
		for(uint i=0;i<ITEMS_PER_WI;i++) {
			if(c+i < over) {
				dtype tmp = x[c+i + x_];
				if(tmp > val) {
					index = c+i;
					val = tmp;
				}
			}
		}

		uint lid = my_get_local_wg_id();
		reduce_indx[lid] = index;
		reduce_vals[lid] = val;
		barrier();
		for(uint i=WG_SIZE / 2; i > 0 ; i>>=1) {
			if(lid < i) {
				dtype val;
				uint ind;
				dtype vl = reduce_vals[lid];
				uint   il = reduce_indx[lid];
				dtype vr = reduce_vals[lid+i];
				uint   ir = reduce_indx[lid+i];
				if(vl > vr) {
					ind=il;
					val=vl;
				}
				else if(vr > vl) {
					ind=ir;
					val=vr;
				}
				else { // vr == vl
					if(il < ir) {
						ind=il;
						val=vl;
					}
					else {
						ind=ir;
						val=vr;
					}
				}
				reduce_vals[lid] = val;
				reduce_indx[lid] = ind;
			}
			barrier(); 
		}

		uint target_index = reduce_indx[0];
		dtype store = outp[out_];
		for(uint i=0;i<ITEMS_PER_WI;i++) {
			if(c+i < over) {
				dtype val = (c + i == target_index) ? store : 0;
				if(factor == 0)
					dx[c+i + dx_] = val;
				else
					dx[c+i + dx_] = dx[c+i + dx_] * factor + val;
			}
		}
	}
	else
	{
		dtype store = outp[out_] * scale;
		
		for(uint i=0;i<ITEMS_PER_WI;i++) {
			if(c+i < over) {
				if(factor == 0)
					dx[c+i + dx_] = store;
				else
					dx[c+i + dx_] = dx[c+i + dx_] * factor + store;
			}
		}
	}
}
