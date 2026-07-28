#version 450

#include "../common/defs.glsl"

#ifndef itype
#define itype int
#endif

#ifndef POOL_MODE
#define POOL_MODE 0
#endif

#ifndef POOL_H
#define POOL_H 1
#endif
#ifndef POOL_W
#define POOL_W 1
#endif

#ifndef STRIDE_H 
#define STRIDE_H 1
#endif

#ifndef STRIDE_W
#define STRIDE_W 1
#endif

#ifndef PAD_H 
#define PAD_H 0
#endif

#ifndef PAD_W
#define PAD_W 0
#endif

#ifndef COUNT_INCLUDE_PAD
#define COUNT_INCLUDE_PAD 0
#endif

#define START_VAL (POOL_MODE == 0 ? -DTYPE_MAX : dtype(0.0f))
#define REDUCE(a,b) (POOL_MODE == 0 ? max((a),(b)) : ((a) + (b)))
#define NORMALIZE_FULL(x) (POOL_MODE == 0 ? (x) : ((x) * (1.0f / (POOL_H * POOL_W))))
#define NORMALIZE_PARTIAL(x,dr,dc,vdr,vdc) (POOL_MODE == 0 ? (x) : (COUNT_INCLUDE_PAD == 0 ? ((x) * (1.0f /((dr)*(dc)))) : ((x) * (1.0f /((vdr)*(vdc))))) )

#ifndef WG_SIZE
#define WG_SIZE 8
#endif

#define INDEX_MAX_SRC (EXPORT_INDEX == 1)

layout(local_size_x = WG_SIZE, local_size_y = WG_SIZE, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer src_buf { dtype src[]; };
	layout(binding = 1, std430) buffer tgt_buf { dtype tgt[]; };
	#if INDEX_MAX_SRC == 1
		layout(binding = 2, std430) buffer indx_buf { itype indx[]; };
	#endif
#endif

layout(push_constant, std430) uniform pooling
{
	uint BC;
	uint inp_H;
	uint inp_W;
	uint out_H;
	uint out_W;
#if USE_BDA
	dtype_addr_ro src;
#endif
	uint src_offset;
#if USE_BDA
	dtype_addr_rw tgt;
#endif
	uint tgt_offset;
#if INDEX_MAX_SRC == 1
	#if USE_BDA
		itype_addr_rw indx;
	#endif
	uint indx_offset;
#endif                                       
             
};

void main()
{
    uint out_r = get_global_id(0);
    uint out_c = get_global_id(1);
    uint bc = get_global_id(2);
    if(bc >= BC || out_r >= out_H || out_c >= out_W)
        return;

    uint row0 = out_r * STRIDE_H - PAD_H;
    uint col0 = out_c * STRIDE_W - PAD_W;
    uint row1 = row0 + POOL_H;
    uint col1 = col0 + POOL_W;

    uint tgt_ = tgt_offset + bc * out_H * out_W;
    uint src_ = src_offset + bc * inp_H * inp_W;

    dtype val = START_VAL;
    #if INDEX_MAX_SRC == 1
		itype index = -1;
		uint indx_ = indx_offset + bc * out_H * out_W;
    #endif
    
    if(row0 >= 0 && col0 >= 0 && row1 <= inp_H && col1 <= inp_W) {
        src_ += row0 * inp_W + col0;
        // #pragma unroll  
        for(uint dr=0;dr<POOL_H;dr++) {
            // #pragma unroll
            for(uint dc = 0;dc < POOL_W; dc++)
            {
                #if INDEX_MAX_SRC == 1
					dtype tmp = src[dr * inp_W + dc + src_];
					if(tmp > val) {
						index = (row0 + dr) * inp_W + col0 + dc;
						val = tmp;
					}
                #else
					val = REDUCE(val, src[dr * inp_W + dc + src_]);
                #endif
            }
        }
        val = NORMALIZE_FULL(val); 
    }
    else
    {
        // #pragma unroll
        for(uint r=row0;r<row1;r++) {
            // #pragma unroll
            for(uint c=col0;c<col1;c++) {
                dtype loaded_val = (r >= 0 && r<inp_H && c>=0 && c<inp_W) ? src[r*inp_W + c + src_] : START_VAL;
                #if INDEX_MAX_SRC == 1
                if(loaded_val > val) {
                    index = r*inp_W + c;
                    val = loaded_val;
                }
                #else
                val = REDUCE(val,loaded_val);
                #endif
            }
        }
        val = NORMALIZE_PARTIAL(val, min(row1,inp_H) - max(row0,0),
                                     min(col1,inp_W) - max(col0,0),
                                     min(row1,inp_H + PAD_H) - max(-PAD_H,row0),
                                     min(col1,inp_W + PAD_W) - max(-PAD_W,col0)
                                     );
    }
    tgt[out_r * out_W + out_c + tgt_] = val;
    #if INDEX_MAX_SRC == 1
		indx[out_r * out_W + out_c + indx_] = index;
	#endif
}
