#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#include "../common/atomic.glsl"
#include "../common/workgroup.glsl"
#include "defs.glsl"

#define DIM_R 0
#define DIM_C 1
#define DIM_BD 2

#define KERN_PAD ((KERN-1)/2)

#define PATCH_H (PATCH_ROWS + KERN - 1)
#define PATCH_W (PATCH_COLS + KERN - 1)

#if USE_BDA == 0
	layout(binding = 0, std430) buffer inp_buffer { uint inp[]; }; // declared as uint and bitcast as float to make atomic operations easier
	layout(binding = 1, std430) readonly buffer kern_buffer { float kern[]; };
	layout(binding = 2, std430) readonly buffer outp_buffer { float outp[]; };
#endif

layout(push_constant, std430) uniform backward_data_conv
{
	uint batch;uint height;uint width;
#if USE_BDA
	__global float *input;
#endif
	uint input_offset;
#if USE_BDA
	__global const float *kern;
#endif
	uint kernel_offset;
#if USE_BDA
	__global float const *output;
#endif
	uint output_offset;
};

void main()
{
    uint inp_offset = input_offset;
    uint outp_offset = output_offset;
    uint kern_offset = kernel_offset;
    uint b = get_global_id(DIM_BD) / CHANNELS;
    uint d = get_global_id(DIM_BD) % CHANNELS;
    uint r = get_global_id(DIM_R) * PATCH_ROWS;
    uint c = get_global_id(DIM_C) * PATCH_COLS;

    if(r >= height || c >= width || b >= batch)
        return;

    kern_offset += (d * KERN * KERN);
    
    inp_offset += (b * CHANNELS * width * height + d * width * height + (r - KERN_PAD) * width + c - KERN_PAD);
    outp_offset += (b * CHANNELS * width * height + d * width * height + r * width + c);

    float K_vals[KERN][KERN];
    float I_vals[PATCH_H][PATCH_W];

    // #pragma unroll
    for(uint dr=0;dr < KERN;dr++)
        // #pragma unroll
        for(uint dc=0;dc<KERN;dc++)
        {
			// may want to come back and make sure this is right later
            K_vals[dr][dc] = kern[kern_offset];
			kern_offset += 1;
		}

            


    //#pragma unroll
    for(uint dr=0;dr<PATCH_ROWS;dr++) {
        if(r+dr >= height)
            break;
        // #pragma unroll
        for(uint dc=0;dc<PATCH_COLS;dc++) {
            if(c+dc>=width)
                break;
            float val = outp[dr*width+dc + outp_offset];
            //#pragma unroll
            for(uint drk=0;drk < KERN;drk++)
                //#pragma unroll
                for(uint dck=0;dck<KERN;dck++)
                    I_vals[dr+drk][dc+dck] = fma(K_vals[drk][dck],val,I_vals[dr+drk][dc+dck]);

        }
    }

    uint y = r-KERN_PAD;
    //#pragma unroll
    for(uint dr=0;dr<PATCH_H;dr++,y++) {
        if(y < 0 || y >= height) {
            //#pragma unroll
            for(uint dc=0;dc<PATCH_W;dc++)
                I_vals[dr][dc]=0;
        }
        if(!(y < 0 || y >= height)) {
            uint x = c - KERN_PAD;
            //#pragma unroll
            for(uint dc=0;dc<PATCH_W;dc++,x++) {
                if(0 <= x && x < width) {
                    #if KERN == 1
                    inp[dr*width+dc + inp_offset] += I_vals[dr][dc];
                    #else
                    atomic_addf(inp_offset + (dr*width+dc), inp,I_vals[dr][dc]);
                    #endif
                }
            }
        }
    }
}
