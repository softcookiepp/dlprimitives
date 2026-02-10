#version 450

#include "../common/defs.glsl"

#if USE_BDA == 0
	layout(binding = 0, std430) buffer src_buf { dtype_src src[]; };
	layout(binding = 1, std430) buffer tgt_buf { dtype_tgt tgt[]; };
#endif

layout(push_constant, std430) uniform copy
{
	uint d0;uint s0;uint t0;
#if DIMS >= 2
	uint d1;uint s1;uint t1;
#endif
#if DIMS >= 3
	uint d2;uint s2;uint t2;
#endif          
#if DIMS >= 4
	uint d3;uint s3;uint t3;
#endif          
#if DIMS >= 5
	uint d4;uint s4;uint t4;
#endif
#if DIMS >= 6
	uint d5;uint s5;uint t5;
#endif
#if DIMS >= 7
	uint d6;uint s6;uint t6;
#endif
#if DIMS >= 8
	uint d7;uint s7;uint t7;
#endif
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
        #if DIMS == 1
            uint i0 = get_global_id(0);
            if(i0 >= d0)
                return;

            tgt[i0*t0 + tgt_offset] = src[i0*s0 + src_offset];
        #elif DIMS == 2
            uint i1 = get_global_id(0);
            uint i0 = get_global_id(1);
            if(i0 >= d0)
                return;
            if(i1 >= d1)
                return;
            tgt[i0*t0 + i1*t1 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + src_offset]);
        #elif DIMS == 3
            uint i2 = get_global_id(0);
            uint i1 = get_global_id(1);
            uint i0 = get_global_id(2);
            if(i0 >= d0)
                return;
            if(i1 >= d1)
                return;
            if(i2 >= d2)
                return;
            tgt[i0*t0 + i1*t1 + i2*t2 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + src_offset]);
        #elif DIMS == 4
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
            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + src_offset]);
        #elif DIMS == 5
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
            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + src_offset]);
        #elif DIMS == 6
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

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + src_offset]);
        #elif DIMS == 7
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

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6 + src_offset]);
        #elif DIMS == 8
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

            tgt[i0*t0 + i1*t1 + i2*t2 + i3*t3 + i4*t4 + i5*t5 + i6*t6 + i7*t7 + tgt_offset] = dtype_tgt(src[i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5 + i6*s6 + i7*s7 + src_offset]);
        #else
        #error "Unsupported dims"
        #endif
}


