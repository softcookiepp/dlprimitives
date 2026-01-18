#version 450
///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include "../common/defs.glsl"
#include "../common/workgroup.glsl"

#ifndef ELTOP
#define ELTOP 0
#endif

#if ELTOP == 0
#define EOP(x,y) ((x) + (y))
#elif ELTOP == 1
#define EOP(x,y) ((x) * (y))
#elif ELTOP == 2
#define EOP(x,y) max((x),(y))
#else
#error "Invaid operation"
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer a_buf { dtype a[]; };
	layout(binding = 1, std430) buffer da_buf { dtype da[]; };
	layout(binding = 2, std430) readonly buffer b_buf { dtype b[]; };
	layout(binding = 3, std430) buffer db_buf { dtype db[]; };
	layout(binding = 4, std430) readonly buffer c_buf { dtype c[]; };
	layout(binding = 5, std430) readonly buffer dc_buf { dtype dc[]; };
#endif

layout(push_constant, std430) uniform eltwise_bwd
{
	uint size;
	uint da_db_select;
#if USE_BDA
	__global const dtype *a;
#endif
	uint  a_offset;
#if USE_BDA
	__global dtype *da;
#endif
	uint  da_offset;
#if USE_BDA
	__global const dtype *b;
#endif
	uint  b_offset;
#if USE_BDA
	__global dtype *db;
#endif
	uint  db_offset;
#if USE_BDA
	__global const dtype *c;
#endif
	uint  c_offset;
#if USE_BDA
	__global const dtype *dc;
#endif
	uint  dc_offset;
	dtype c1;
	dtype c2;
	dtype factor_a;
	dtype factor_b;
};

void main()
{
    uint pos = get_global_id(0);
    if(pos >= size)
        return;
    dtype dy = ACTIVATION_FINV(c[pos + c_offset], dc[pos + dc_offset]);
    if((da_db_select & 1) > 0) { // da+a
        dtype da_val;
        #if ELTOP == 0
        da_val = dy * c1;
        #elif ELTOP == 1
        da_val = dy * c1 * c2 * b[pos + b_offset];
        #elif ELTOP == 2
        if(c1*a[pos + a_offset] >= c2*b[pos + b_offset]) 
            da_val = c1 * dy;
        else
            da_val = 0;
        #endif
        if(factor_a == 0)
            da[pos + da_offset] = da_val;
        else
            da[pos + da_offset] = da[pos + da_offset] * factor_a + da_val;
    }
    if( (da_db_select & 2) > 0) { // db+b
        dtype db_val;
        #if ELTOP == 0
        db_val = dy * c2;
        #elif ELTOP == 1
        db_val = dy * c1 * c2 * a[pos + a_offset];
        #elif ELTOP == 2
        if(c1*a[pos + a_offset] >= c2*b[pos + b_offset])
            db_val = 0;
        else
            db_val = c2 * dy;
        #endif
        if(factor_b == 0)
            db[pos + db_offset] = db_val;
        else
            db[pos + db_offset] = db[pos + db_offset] * factor_b + db_val;
    }
}

