#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"

#ifndef DTYPE_SRC
	#define DTYPE_SRC DTYPE
#endif

#if DTYPE_SRC == DTYPE_F16
	#define dtype_src float16_t
#elif DTYPE_SRC == DTYPE_F32
	#define dtype_src float
#elif DTYPE_SRC == DTYPE_F64
	#define dtype_src double
#elif DTYPE_SRC == DTYPE_I8
	#define dtype_src int8_t
#elif DTYPE_SRC == DTYPE_I16
	#define dtype_src int16_t
#elif DTYPE_SRC == DTYPE_I32
	#define dtype_src int
#elif DTYPE_SRC == DTYPE_I64
	#define dtype_src int64_t
#elif DTYPE_SRC == DTYPE_U8
	#define dtype_src uint8_t
#elif DTYPE_SRC == DTYPE_U16
	#define dtype_src uint16_t
#elif DTYPE_SRC == DTYPE_U32
	#define dtype_src uint
#elif DTYPE_SRC == DTYPE_U64
	#define dtype_src uint64_t	
#endif

#ifndef DTYPE_TGT
	#define DTYPE_TGT DTYPE
#endif

#if DTYPE_TGT == DTYPE_F16
	#define dtype_tgt float16_t
#elif DTYPE_TGT == DTYPE_F32
	#define dtype_tgt float
#elif DTYPE_TGT == DTYPE_F64
	#define dtype_tgt double
#elif DTYPE_TGT == DTYPE_I8
	#define dtype_tgt int8_t
#elif DTYPE_TGT == DTYPE_I16
	#define dtype_tgt int16_t
#elif DTYPE_TGT == DTYPE_I32
	#define dtype_tgt int
#elif DTYPE_TGT == DTYPE_I64
	#define dtype_tgt int64_t
#elif DTYPE_TGT == DTYPE_U8
	#define dtype_tgt uint8_t
#elif DTYPE_TGT == DTYPE_U16
	#define dtype_tgt uint16_t
#elif DTYPE_TGT == DTYPE_U32
	#define dtype_tgt uint
#elif DTYPE_TGT == DTYPE_U64
	#define dtype_tgt uint64_t	
#endif

layout(constant_id = 3) const uint DIMS = 1;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer src_buf { dtype_src src[]; };
	layout(binding = 1, std430) buffer tgt_buf { dtype_tgt tgt[]; };
#endif

layout(push_constant, std430) uniform copy
{
	Shape shape;
	Shape srcStride;
	Shape tgtStride;
	#if USE_BDA
		__global dtype_src const *src;
	#endif
	uint src_offset;
	#if USE_BDA
		__global dtype_tgt *tgt;
	#endif
	uint tgt_offset;
};

void main()
{
	uint srcIdx;
	uint tgtIdx;
	
	Shape pos = getPosFromTriIndex(gl_GlobalInvocationID, shape, DIMS);
	if (!posValid(shape, pos, DIMS)) return;
	srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
	tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
	tgt[tgtIdx + tgt_offset] = dtype_tgt(src[src_offset + srcIdx]);
}


