#version 450

#ifndef REDUCE_DIMS
#error "REDUCE_DIMS must be defined"
#endif
#ifndef DIMS
#error "DIMS must be defined"
#endif

#include "../common/broadcast_dims.glsl"
#if SMALL_REDUCTION
	#include "../common/workgroup.glsl"
#else
	layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

#define NORMAL_DIMS (DIMS - REDUCE_DIMS)

#if REDUCE_DIMS > DIMS
#error "REDUCE_DIMS must be <= DIMS"
#endif
#if REDUCE_DIMS < 0
#error "Need at least 1 dim for reduction"
#endif

#ifndef SMALL_REDUCTION
#define SMALL_REDUCTION 0
#endif

#ifndef TWO_STAGE_REDUCTION
#define TWO_STAGE_REDUCTION 0
#endif


uint get_base_offset(Shape s, Shape strides, uint offset)
{
    uint r = offset;
    //#pragma unroll
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
#if REDUCE_DIMS == 0
	#define next_pos(limits, pos) {} // nothing
#elif REDUCE_DIMS == 1
	#define next_pos(limits, pos) { pos[0] += 1; }
#elif REDUCE_DIMS == 2
	#define next_pos(limits, pos) \
	{ \
		pos.s[1] += 1; \
		if (pos.s[1] == limits.s[1]) \
		{ \
			pos.s[1] = 0; \
			pos.s[0] += 1; \
		} \
	}
#elif REDUCE_DIMS == 3
	#define next_pos(limits, pos) \
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
#else
	#error "Too many reduction dims"
#endif

#if REDUCE_DIMS >= 1
Shape get_pos(Shape limits, uint reduce_item)
{
    Shape r;
#if REDUCE_DIMS == 1
    r.s[0] = reduce_item;
#elif REDUCE_DIMS == 2
    r.s[0] = reduce_item / limits.s[1];
    r.s[1] = reduce_item % limits.s[1];
#elif REDUCE_DIMS == 3
    r.s[2] = reduce_item % limits.s[2];
    uint ri2 = reduce_item / limits.s[2];
    r.s[1] = ri2 % limits.s[1];
    r.s[0] = ri2 / limits.s[1];
#else 
// for total dims limit = 5 shouldn't be more than 3 reduction dims otherwise they will be shrinked
#error "Too many reduction dims"
#endif

#if NORMAL_DIMS == 0
    // nothing
#elif NORMAL_DIMS == 1
    r.s[REDUCE_DIMS + 0] = get_global_id(1);
#elif NORMAL_DIMS == 2
    r.s[REDUCE_DIMS + 0] = get_global_id(2);      
    r.s[REDUCE_DIMS + 1] = get_global_id(1);      
#elif NORMAL_DIMS == 3
    r.s[REDUCE_DIMS + 0] = get_global_id(2) / limits.s[REDUCE_DIMS+1];      
    r.s[REDUCE_DIMS + 1] = get_global_id(2) % limits.s[REDUCE_DIMS+1];      
    r.s[REDUCE_DIMS + 2] = get_global_id(1);      
#elif NORMAL_DIMS == 4
    r.s[REDUCE_DIMS + 0] = get_global_id(2) / limits.s[REDUCE_DIMS+1];      
    r.s[REDUCE_DIMS + 1] = get_global_id(2) % limits.s[REDUCE_DIMS+1];      
    r.s[REDUCE_DIMS + 2] = get_global_id(1) / limits.s[REDUCE_DIMS+3];
    r.s[REDUCE_DIMS + 3] = get_global_id(1) % limits.s[REDUCE_DIMS+3];
#else
#error "Unsupported dim"
#endif
    return r;
}

#endif

bool valid_save_pos(Shape pos, Shape limits)
{
    // #pragma unroll
    for(uint i=REDUCE_DIMS;i<DIMS;i++)
        if(pos.s[i] >= limits.s[i])
            return 0;
    return 1;

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
	//#define PARAM_INPUT(type,I) ,__global type const *px##I,uint px##I##_offset,Shape xstrides##I
	#define PARAM_INPUT(type,I) layout(set = 0, binding = I, std430) readonly px##I##_buf { type px##I[]; };
	#if TWO_STAGE_REDUCTION == 1
		#define PARAM_OUTPUT(type,ptype,I) ,__global type *py##I,uint py##I##_offset,Shape ystrides##I
	#else
		#define PARAM_OUTPUT(type,ptype,I) ,__global type *py##I,uint py##I##_offset,Shape ystrides##I,ptype alpha##I,ptype beta##I
	#endif
	#define PAPAM_WEIGHT(type, I) ,type w##I
#endif


#define PREPARE_LOAD_INPUT(type,I) \
    uint input_offset_##I = get_base_offset(index,xstrides##I,px##I##_offset); \
    type x##I;

#define LOAD_INPUT(I) x##I = px##I[input_offset_##I + get_reduce_offset(index,xstrides##I)];
#define SAVE_OUTPUT(I) py##I[get_base_offset(index,ystrides##I,py##I##_offset)] = reduce_y##I;

#define my_get_local_wg_id() ((get_local_id(2) * get_local_size(1) * get_local_size(0)) + (get_local_id(1) * get_local_size(0)) + get_local_id(0))

#if SMALL_REDUCTION == 1
#define REDUCE_INIT(type,I) type reduce_y##I,y##I;
#else
#define REDUCE_INIT(type,I) \
    __local type my_reduce_##I[WG_SIZE]; \
    type reduce_y##I,y##I; 
#endif    

#define SAVE_REDUCE(I) my_reduce_##I[lid] = reduce_y##I;
#define LOAD_REDUCE(I) reduce_y##I = my_reduce_##I[lid]; y##I = my_reduce_##I[nxt];

#if SMALL_REDUCTION == 1
#define LOAD_REDUCED_SAVE_GLOBAL(I) \
do { \
    py##I += get_base_offset(index,ystrides##I,py##I##_offset); \
    reduce_y##I *= alpha##I; \
    if(beta##I)  \
        *py##I = beta##I * *py##I + reduce_y##I; \
    else \
        *py##I = reduce_y##I; \
}while(0)
#elif TWO_STAGE_REDUCTION == 0
#define LOAD_REDUCED_SAVE_GLOBAL(I) \
do { \
    y##I = alpha##I * my_reduce_##I[0]; \
    py##I += get_base_offset(index,ystrides##I,py##I##_offset); \
    if(beta##I) \
        *py##I = beta##I * *py##I + y##I; \
    else \
        *py##I = y##I; \
} while(0)
#else //TWO_STAGE_REDUCTION == 1
#define LOAD_REDUCED_SAVE_GLOBAL(I) \
do { \
    py##I += py##I##_offset + get_group_id(0); \
    py##I += reduce_stride * get_base_offset(index,ystrides##I,0); \
    *py##I = my_reduce_##I[0]; \
} while(0)
#endif


__kernel 
void exec(Shape limit 
                   PARAMS
#if TWO_STAGE_REDUCTION == 1
                   ,uint reduce_stride
#endif                                      
                   )

{
#if REDUCE_DIMS == 0
    #if ITEMS_PER_WI > 1
    #error "Invalid Items per wi size"
    #endif
    uint reduce_item = 0;
    Shape index0 = get_pos_broadcast(limit);
#else    
    uint reduce_item = get_global_id(0) * ITEMS_PER_WI;
    Shape index0 = get_pos(limit,reduce_item);
#endif
    Shape index = index0;
    PREPARE_LOAD_INPUT_ALL
    REDUCE_INIT_ALL

    #pragma unroll(8)
    for(uint item=0;item < ITEMS_PER_WI;item++) {
        if(valid_pos(index,limit)) {
            LOAD_INPUT_ALL
            CALC
            REDUCE
        }
#if ITEMS_PER_WI > 1
        next_pos(limit,&index);
        reduce_item ++;
#endif
    }

    #if SMALL_REDUCTION == 0

    uint lid = get_local_id(0); 

    SAVE_REDUCE_ALL
    
    barrier(CLK_LOCAL_MEM_FENCE); 
    for(uint i= WG_SIZE / 2;i>0; i>>= 1) { 
        if(lid < i) { 
            uint nxt = lid+i;
            LOAD_REDUCE_ALL
            REDUCE
            SAVE_REDUCE_ALL
        } 
        barrier(CLK_LOCAL_MEM_FENCE); 
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


