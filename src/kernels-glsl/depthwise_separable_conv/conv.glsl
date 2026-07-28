#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#include "../common/activation.glsl"
#include "defs.glsl"
//#include "../common/atomic.glsl"

PREPARE_ACTIVATION(0)

#define DIM_R 0
#define DIM_C 1
#define DIM_BD 2

#define KERN_PAD ((KERN-1)/2)

#define PATCH_H (PATCH_ROWS + KERN - 1)
#define PATCH_W (PATCH_COLS + KERN - 1)

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { float inp[]; };
	layout(binding = 1, std430) readonly buffer kern_buf { float kern[]; };
	layout(binding = 2, std430) readonly buffer bias_buf { float bias[]; };
	layout(binding = 3, std430) buffer outp_buf { float outp[]; };
#endif

layout(push_constant, std430) uniform conv
{
	uint batch;
	uint height;
	uint width;
#if USE_BDA
	__global const float *input;
#endif
	uint input_offset;
#if USE_BDA
	__global const float *kern;
#endif
	uint kernel_offset;
#if BIAS != 0
#if USE_BDA
	__global const float *bias;
#endif
	uint bias_offset;
#endif          
#if USE_BDA
	__global float *output;
#endif
	uint output_offset;
	float scale;
};
          
void main()
{
    uint inp_offset = input_offset;
    uint outp_offset = output_offset;
    uint kern_offset = kernel_offset;
    uint r = get_global_id(DIM_R) * PATCH_ROWS;
    uint c = get_global_id(DIM_C) * PATCH_COLS;
    uint b = get_global_id(DIM_BD) / CHANNELS;
    uint d = get_global_id(DIM_BD) % CHANNELS;

    if(r >= height || c >= width || b >= batch)
        return;

    kern_offset += (d * KERN * KERN);
    
    inp_offset  += (b * CHANNELS * width * height + d * width * height + (r - KERN_PAD) * width + c - KERN_PAD);
    outp_offset += (b * CHANNELS * width * height + d * width * height + r * width + c);

    float K_vals[KERN][KERN];
    float I_vals[PATCH_H][PATCH_W];

    // #pragma unroll
    for(uint dr=0;dr < KERN;dr++)
        // #pragma unroll
        for(uint dc=0;dc<KERN;dc++)
        {
            K_vals[dr][dc] = kern[kern_offset];
            kern_offset += 1;
		}

    #if BIAS != 0
        float start_val = bias[bias_offset + d];
    #else
        const float start_val = 0;
    #endif

            

    uint y = r-KERN_PAD;
    // #pragma unroll
    for(uint dr=0;dr<PATCH_H;dr++,y++) {
        if(y < 0 || y >= height) {
            // #pragma unroll
            for(uint dc=0;dc<PATCH_W;dc++)
                I_vals[dr][dc]=0;
        }
        else {
            uint x = c - KERN_PAD;
            // #pragma unroll
            for(uint dc=0;dc<PATCH_W;dc++,x++) {
                I_vals[dr][dc]=(0 <= x && x < width) ? inp[dr*width+dc + inp_offset] : 0;
            }
        }
    }


    // #pragma unroll
    for(uint dr=0;dr<PATCH_ROWS;dr++) {
        if(r+dr >= height)
            break;
        // #pragma unroll
        for(uint dc=0;dc<PATCH_COLS;dc++) {
            if(c+dc>=width)
                break;
            float sum = start_val;
            // #pragma unroll
            for(uint drk=0;drk < KERN;drk++)
                // #pragma unroll
                for(uint dck=0;dck<KERN;dck++)
                    sum = fma(K_vals[drk][dck],I_vals[dr+drk][dc+dck],sum);

            float value = ACTIVATION_F(sum);
            uint optr = outp_offset + dr*width+dc;
            if(scale == 0.0)
                outp[optr] = value;
            else
                outp[optr] = scale * outp[optr] + value;
        }
    }

}
