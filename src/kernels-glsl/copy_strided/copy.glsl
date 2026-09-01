#version 450

#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
#include "../common/shape.glsl"

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
	
	Shape pos = getPosFromTriIndex(shape, DIMS);
	if (!posValid(shape, pos, DIMS)) return;
	srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
	tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
	tgt[tgtIdx + tgt_offset] = dtype_tgt(src[src_offset + srcIdx]);
}


