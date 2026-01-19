#version 450
#include "../common/defs.glsl"

#ifndef CHANNELS
	#define CHANNELS 3
#endif
#ifndef KERN_H
	#define KERN_H 8
#endif
#ifndef KERN_W
	#define KERN_W 8
#endif
#ifndef PAD_H
	#define PAD_H 1
#endif
#ifndef PAD_W
	#define PAD_W 1
#endif
#ifndef STRIDE_H
	#define STRIDE_H 1
#endif
#ifndef STRIDE_W
	#define STRIDE_W 1
#endif
#ifndef DILATE_H
	#define DILATE_H 1
#endif
#ifndef DILATE_W
	#define DILATE_W 1
#endif

layout(local_size_x = 1, local_size_y = 8, local_size_z = 8) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer img_buf { dtype img[]; };
	layout(binding = 1, std430) writeonly buffer mat_buf { dtype mat[]; };
#endif

layout(push_constant, std430) uniform im2col
{
	uint batch;
	uint src_rows;
	uint src_cols;
	uint rows;
	uint cols;
#if USE_BDA
	__global dtype const *img;
#endif
	uint  img_offset;
#if USE_BDA
	__global dtype *mat;
#endif
	uint  mat_offset;
};

void main()
{
    uint img_ = img_offset;
    uint mat_ = mat_offset;

    uint gid = get_global_id(0);
    
    uint chan = gid % CHANNELS;
    uint b    = gid / CHANNELS;
    uint r    = get_global_id(1);
    uint c    = get_global_id(2);

    if(r >= rows || c >= cols || chan >= CHANNELS || b >= batch)
        return;
    mat_ += CHANNELS * (KERN_H * KERN_W) * rows * cols * b;
    img_ += CHANNELS * src_rows * src_cols * b;
    uint mat_row = r * cols + c;
    uint mat_col = chan * (KERN_H * KERN_W);
    mat_ += mat_row * (CHANNELS * KERN_H * KERN_W) + mat_col;
    uint y_pos = -PAD_H + r * STRIDE_H;
    uint x_pos = -PAD_W + c * STRIDE_W;
    img_ += src_cols * (chan * src_rows + y_pos) + x_pos;

    #if PAD_H == 0 && PAD_W == 0
    // #pragma unroll
    for(uint dy = 0;dy < KERN_H * DILATE_H ;dy+= DILATE_H, img_ += src_cols * DILATE_H) {
        // #pragma unroll
        for(uint dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W)
        {
            mat[mat_] = img[dx + img_];
            mat_ += 1;
        }
    }
    #else
    // #pragma unroll
    for(uint dy = 0;dy < KERN_H * DILATE_H ;dy+= DILATE_H, img_ += src_cols * DILATE_H) {
        uint y = y_pos + dy;
        // #pragma unroll
        for(uint dx=0;dx < KERN_W * DILATE_W ;dx+= DILATE_W) {
            uint x = x_pos + dx;
            mat[mat_] = (y>= 0 && y < src_rows && x >= 0 && x < src_cols) ? img[dx + img_] : 0;
            mat_ += 1;
        }
    }
    #endif
}
