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
	uint d0;uint s0;uint t0;
	uint d1;uint s1;uint t1;
	uint d2;uint s2;uint t2;
	uint d3;uint s3;uint t3;
	uint d4;uint s4;uint t4;
	uint d5;uint s5;uint t5;
	uint d6;uint s6;uint t6;
	uint d7;uint s7;uint t7;
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
	//src+=src_offset;
	//tgt+=tgt_offset;
	
	Shape shape;
	shape.s[0] = d0;
	shape.s[1] = d1;
	shape.s[2] = d2;
	shape.s[3] = d3;
	shape.s[4] = d4;
	shape.s[5] = d5;
	shape.s[6] = d6;
	shape.s[7] = d7;
	
	Shape tgtStride;
	tgtStride.s[0] = t0;
	tgtStride.s[1] = t1;
	tgtStride.s[2] = t2;
	tgtStride.s[3] = t3;
	tgtStride.s[4] = t4;
	tgtStride.s[5] = t5;
	tgtStride.s[6] = t6;
	tgtStride.s[7] = t7;
	
	Shape srcStride;
	srcStride.s[0] = s0;
	srcStride.s[1] = s1;
	srcStride.s[2] = s2;
	srcStride.s[3] = s3;
	srcStride.s[4] = s4;
	srcStride.s[5] = s5;
	srcStride.s[6] = s6;
	srcStride.s[7] = s7;
	
	uint srcIdx;
	uint tgtIdx;
	
	Shape pos; // this gets used later, as the kernel has its own wacky way of getting positions
	#if 1
		pos = getPosFromTriIndex(shape, DIMS);
		if (!posValid(shape, pos, DIMS)) return;
		srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
		tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
		tgt[tgtIdx + tgt_offset] = dtype_tgt(src[src_offset + srcIdx]);
	#else
		if (DIMS == 1)
		{
			#if 1
				uint i0 = get_global_id(0);
				if(i0 >= d0)
					return;
				
				srcIdx = getStridedIndex(i0, shape, srcStride, DIMS);
				tgtIdx = getStridedIndex(i0, shape, tgtStride, DIMS);

				tgt[tgtIdx + tgt_offset] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				uint i0 = get_global_id(0);
				if(i0 >= d0)
					return;

				tgt[i0*t0 + tgt_offset] = src[i0*s0 + src_offset];
			#endif
		}
		else if (DIMS == 2)
		{
			uint i1 = get_global_id(0);
			uint i0 = get_global_id(1);
			if(i0 >= d0)
				return;
			if(i1 >= d1)
				return;
			pos.s[0] = i0;
			pos.s[1] = i1;
			srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
			tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
			#if 1
				tgt[tgtIdx + tgt_offset] = dtype_tgt(src[src_offset + srcIdx]);
			#else
				tgt[i0*t0 + i1*t1 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + src_offset]);
			#endif
		}
		else if (DIMS == 3)
		{
			uint i2 = get_global_id(0);
			uint i1 = get_global_id(1);
			uint i0 = get_global_id(2);
			if(i0 >= d0)
				return;
			if(i1 >= d1)
				return;
			if(i2 >= d2)
				return;
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + src_offset]);
			#endif
		}
		else if (DIMS == 4)
		{
			uint ic = get_global_id(0);
			uint i1 = get_global_id(1);
			uint i0 = get_global_id(2);
			if(i0 >= d0)
				return;
			if(i1 >= d1)
				return;
			if(ic >= d2*d3)
				return;
			uint i2 = ic / d3;
			uint i3 = ic % d3;
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			pos.s[3] = i3;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + src_offset]);
			#endif
		}
		else if (DIMS == 5)
		{
			uint i34 = get_global_id(0);
			uint i12 = get_global_id(1);
			uint i0 = get_global_id(2);
			if(i0 >= d0)
				return;
			if(i12 >= d1*d2)
				return;
			if(i34 >= d3*d4)
				return;
			uint i1  = i12 / d2;
			uint i2  = i12 % d2;
			uint i3  = i34 / d4;
			uint i4  = i34 % d4;
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			pos.s[3] = i3;
			pos.s[4] = i4;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + src_offset]);
			#endif
		}
		else if (DIMS == 6)
		{
			uint i45 = get_global_id(0);
			uint i23 = get_global_id(1);
			uint i01 = get_global_id(2);
			if(i01 >= d0*d1)
				return;
			if(i23 >= d2*d3)
				return;
			if(i45 >= d4*d5)
				return;
			uint i0  = i01 / d1;
			uint i1  = i01 % d1;
			uint i2  = i23 / d3;
			uint i3  = i23 % d3;
			uint i4  = i45 / d5;
			uint i5  = i45 % d5;
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			pos.s[3] = i3;
			pos.s[4] = i4;
			pos.s[5] = i5;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + src_offset]);
			#endif
		}
		else if (DIMS == 7)
		{
			uint i456 = get_global_id(0);
			uint i23  = get_global_id(1);
			uint i01  = get_global_id(2);
			if(i01 >= d0*d1)
				return;
			if(i23 >= d2*d3)
				return;
			if(i456 >= d4*d5*d6)
				return;
			uint i0  = i01 / d1;
			uint i1  = i01 % d1;
			uint i2  = i23 / d3;
			uint i3  = i23 % d3;
			uint i4  = i456 / (d5*d6);
			uint i56 = i456 % (d5*d6);
			uint i5  = i56 / d6;
			uint i6  = i56 % d6;
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			pos.s[3] = i3;
			pos.s[4] = i4;
			pos.s[5] = i5;
			pos.s[6] = i6;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6 + src_offset]);
			#endif
		}
		else if (DIMS == 8)
		{
			uint i567 = get_global_id(0);
			uint i234 = get_global_id(1);
			uint i01  = get_global_id(2);
			if(i01 >= d0*d1)
				return;
			if(i234 >= d2*d3*d4)
				return;
			if(i567 >= d5*d6*d7)
				return;
			uint i0  = i01 / d1;
			uint i1  = i01 % d1;

			uint i2  = i234 / (d3*d4);
			uint i34 = i234 % (d3*d4);
			uint i3  = i34 / d4;
			uint i4  = i34 % d4;

			uint i5  = i567 / (d6*d7);
			uint i67 = i567 % (d6*d7);
			uint i6  = i67 / d7;
			uint i7  = i67 % d7;
			
			pos.s[0] = i0;
			pos.s[1] = i1;
			pos.s[2] = i2;
			pos.s[3] = i3;
			pos.s[4] = i4;
			pos.s[5] = i5;
			pos.s[6] = i6;
			pos.s[7] = i7;
			#if 1
				srcIdx = getStridedIndexFromPos(pos, srcStride, DIMS);
				tgtIdx = getStridedIndexFromPos(pos, tgtStride, DIMS);
				tgt[tgt_offset + tgtIdx] = dtype_tgt(src[srcIdx + src_offset]);
			#else
				tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6 + i7*t7 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6 + i7*s7 + src_offset]);
			#endif
		}
	#endif
}


