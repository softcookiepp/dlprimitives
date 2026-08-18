#include "../common/defs.glsl"
#include "../common/workgroup.glsl"
layout(constant_id = 3) const uint REDUCE_DIMS = 1;
layout(constant_id = 4) const uint DIMS = 1;
layout(constant_id = 5) const uint ITEMS_PER_WI = 1;
#include "../common/broadcast_dims.glsl"

#define NORMAL_DIMS (DIMS - REDUCE_DIMS)

#ifndef SMALL_REDUCTION
	#define SMALL_REDUCTION 0
#endif

#ifndef TWO_STAGE_REDUCTION
	#define TWO_STAGE_REDUCTION 0
#endif


uint get_base_offset(Shape s, Shape strides, uint offset)
{
    uint r = offset;
    UNROLL(REDUCE_DIMS)
    for(uint i=REDUCE_DIMS;i<DIMS;i++) {
        r+= s.s[i]*strides.s[i];
    }
    return r;
}

uint get_reduce_offset(Shape s, Shape strides)
{
    uint r = 0;
    // #pragma unroll
    for(uint i=0;i<REDUCE_DIMS;i++) {
        r+= s.s[i]*strides.s[i];
    }
    return r;
}

// because GLSL doesn't support local pointers :c
#define next_pos(limits, pos) \
if (REDUCE_DIMS == 0) {} \
else if (REDUCE_DIMS == 1) { pos.s[0] += 1; } \
else if (REDUCE_DIMS == 2) \
{ \
	pos.s[1] += 1; \
	if (pos.s[1] == limits.s[1]) \
	{ \
		pos.s[1] = 0; \
		pos.s[0] += 1; \
	} \
} \
else if (REDUCE_DIMS == 3) \
{ \
	pos.s[2]++; \
	if(pos.s[2] == limits.s[2]) \
	{ \
		pos.s[2] = 0; \
		pos.s[1] ++; \
		if(pos.s[1] == limits.s[1]) \
		{ \
			pos.s[1] = 0; \
			pos.s[0] ++; \
		} \
	} \
}

//#if REDUCE_DIMS >= 1
Shape get_pos(Shape limits, uint reduce_item)
{
    Shape r;
	if (REDUCE_DIMS == 1)
	{
		r.s[0] = reduce_item;
	}
	else if (REDUCE_DIMS == 2)
	{
		r.s[0] = reduce_item / limits.s[1];
		r.s[1] = reduce_item % limits.s[1];
	}
	else if (REDUCE_DIMS == 3)
	{
		r.s[2] = reduce_item % limits.s[2];
		uint ri2 = reduce_item / limits.s[2];
		r.s[1] = ri2 % limits.s[1];
		r.s[0] = ri2 / limits.s[1];
	}
	// for total dims limit = 5 shouldn't be more than 3 reduction dims otherwise they will be shrinked
	
	if (NORMAL_DIMS == 0)
	{
		// nothing
	}
	else if (NORMAL_DIMS == 1)
	{
		r.s[REDUCE_DIMS + 0] = get_global_id(1);
	}
	else if (NORMAL_DIMS == 2)
	{
		r.s[REDUCE_DIMS + 0] = get_global_id(2);      
		r.s[REDUCE_DIMS + 1] = get_global_id(1);
	} 
	else if (NORMAL_DIMS == 3)
	{
		r.s[REDUCE_DIMS + 0] = get_global_id(2) / limits.s[REDUCE_DIMS+1];      
		r.s[REDUCE_DIMS + 1] = get_global_id(2) % limits.s[REDUCE_DIMS+1];      
		r.s[REDUCE_DIMS + 2] = get_global_id(1);
	}
	else if (NORMAL_DIMS == 4)
	{
		r.s[REDUCE_DIMS + 0] = get_global_id(2) / limits.s[REDUCE_DIMS+1];      
		r.s[REDUCE_DIMS + 1] = get_global_id(2) % limits.s[REDUCE_DIMS+1];      
		r.s[REDUCE_DIMS + 2] = get_global_id(1) / limits.s[REDUCE_DIMS+3];
		r.s[REDUCE_DIMS + 3] = get_global_id(1) % limits.s[REDUCE_DIMS+3];
	}
    return r;
}

//#endif

bool valid_save_pos(Shape pos, Shape limits)
{
    // #pragma unroll
    for(uint i=REDUCE_DIMS;i<DIMS;i++)
        if(pos.s[i] >= limits.s[i])
            return false;
    return true;

}


bool valid_pos(Shape pos,Shape limits)
{
    // #pragma unroll
    for(uint i=0;i<DIMS;i++)
        if(pos.s[i] >= limits.s[i])
            return false;
    return true;

}

#if USE_BDA
	#error "not implemented"
#else
	#define PARAM_INPUT_BUF(type, I, bind) layout(binding = bind, std430) readonly buffer px##I##_buf { type px##I[]; };
	#define PARAM_INPUT_BUF_OFFSET(type, I) uint px##I##_p = 0; 
	#define PARAM_INPUT(type,I) uint px##I##_offset; Shape xstrides##I;
	
	#define PARAM_OUTPUT_BUF_OFFSET(type, ptype, I) uint py##I##_p = 0;
	#define PARAM_OUTPUT_BUF(type, ptype, I, bind) layout(binding = bind) buffer py##I##_buf { type py##I[]; };
	#if TWO_STAGE_REDUCTION == 1 // TODO: make this file a template and split into multiple files
		#define PARAM_OUTPUT(type, ptype, I) uint py##I##_offset; Shape ystrides##I;
	#else
		#define PARAM_OUTPUT(type, ptype, I) uint py##I##_offset; Shape ystrides##I; ptype alpha##I; ptype beta##I;
	#endif
	#define PARAM_WEIGHT(type, I) type w##I;
#endif


#define PREPARE_LOAD_INPUT(type,I) \
    uint input_offset_##I = get_base_offset(index,xstrides##I,px##I##_offset); \
    type x##I;

#define LOAD_INPUT(I) x##I = px##I[input_offset_##I + get_reduce_offset(index,xstrides##I) + px##I##_p];
#define SAVE_OUTPUT(I) py##I[get_base_offset(index,ystrides##I,py##I##_offset) + px##I##_p] = reduce_y##I;

#define my_get_local_wg_id() ((get_local_id(2) * get_local_size(1) * get_local_size(0)) + (get_local_id(1) * get_local_size(0)) + get_local_id(0))

#if SMALL_REDUCTION == 1
	#define REDUCE_INIT(type,I) type reduce_y##I, y##I
#else
	#define REDUCE_INIT(type,I) shared type my_reduce_##I[localSizeX]; type reduce_y##I, y##I 
#endif    

#define SAVE_REDUCE(I) my_reduce_##I[lid] = reduce_y##I;
#define LOAD_REDUCE(I) reduce_y##I = my_reduce_##I[lid]; y##I = my_reduce_##I[nxt];

#if SMALL_REDUCTION == 1
	#define LOAD_REDUCED_SAVE_GLOBAL(I) \
	do { \
		py##I##_p += get_base_offset(index,ystrides##I,py##I##_offset); \
		reduce_y##I *= alpha##I; \
		if(bool(beta##I))  \
			py##I[py##I##_p] = beta##I * py##I[py##I##_p] + reduce_y##I; \
		else \
			py##I[py##I##_p] = reduce_y##I; \
	}while(false);
#elif TWO_STAGE_REDUCTION == 0
	#define LOAD_REDUCED_SAVE_GLOBAL(I) \
	do { \
		y##I = alpha##I * my_reduce_##I[0]; \
		py##I##_p += get_base_offset(index,ystrides##I,py##I##_offset); \
		if(bool(beta##I)) \
			py##I[py##I##_p] = beta##I * py##I[py##I##_p] + y##I; \
		else \
			py##I[py##I##_p] = y##I; \
	} while(false);
#else //TWO_STAGE_REDUCTION == 1
#define LOAD_REDUCED_SAVE_GLOBAL(I) \
	do { \
		py##I##_p += py##I##_offset + get_group_id(0); \
		py##I##_p += reduce_stride * get_base_offset(index,ystrides##I,0); \
		py##I[py##I##_p] = my_reduce_##I[0]; \
	} while(false);
#endif

#if USE_BDA
	#error "not implemented!"
#else
	BUFFER_DEFS
#endif

layout(push_constant, std430) uniform exec
{
	Shape limit;
	PARAMS
#if TWO_STAGE_REDUCTION == 1
	uint reduce_stride;
#endif                                      
};

REDUCE_INIT_SHARED

void exec_impl()
{
	BUFFER_OFFSETS
	uint reduce_item;
	Shape index0;
	if (REDUCE_DIMS == 0)
	{
		#if 0 // we are folding the constants, no longer do we have any use for this
			#if ITEMS_PER_WI > 1
				#error "Invalid Items per wi size"
			#endif
		#endif
		reduce_item = 0;
		index0 = get_pos_broadcast(limit);
	}
	else
	{
		reduce_item = get_global_id(0) * ITEMS_PER_WI;
		index0 = get_pos(limit,reduce_item);
	}
    Shape index = index0;
    PREPARE_LOAD_INPUT_ALL
    REDUCE_INIT_ALL

    UNROLL(ITEMS_PER_WI)
    for(uint item=0;item < ITEMS_PER_WI;item++) {
        if(valid_pos(index,limit)) {
            LOAD_INPUT_ALL
            CALC
            REDUCE
        }
		if (ITEMS_PER_WI > 1)
		{
			next_pos(limit, index);
			reduce_item ++;
		}
    }

    #if SMALL_REDUCTION == 0

		uint lid = get_local_id(0); 

		SAVE_REDUCE_ALL
		
		barrier(); 
		for(uint i= localSizeX / 2;i>0; i>>= 1) { 
			if(lid < i) { 
				uint nxt = lid+i;
				LOAD_REDUCE_ALL
				REDUCE
				SAVE_REDUCE_ALL
			} 
			barrier(); 
		} 
		if(lid == 0) {
			if(valid_save_pos(index0,limit)) {
				LOAD_REDUCED_SAVE_GLOBAL_ALL
			}
		}

    #else
		if(valid_save_pos(index0,limit)) {
			LOAD_REDUCED_SAVE_GLOBAL_ALL
		}
    #endif
}


