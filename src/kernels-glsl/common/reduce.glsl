///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#ifndef CUSTOM_REDUCE
#define CUSTOM_REDUCE 0
#endif

#ifndef my_get_local_wg_id
#define my_get_local_wg_id() ((get_local_id(2) * get_local_size(1) * get_local_size(0)) + (get_local_id(1) * get_local_size(0)) + get_local_id(0))
#endif

// this is incompatible with GLSL
#define REDUCE_PREPARE(wg_size, dtype) shared dtype my_reduce[wg_size]
#define REDUCE_FILL(value) \
{ \
	uint lid = my_get_local_wg_id(); \
	my_reduce[lid] = value; \
	barrier(); \
}

#define REDUCE_USING_OP(myval,reduce_op, wg_size) \
    do { \
        uint lid = my_get_local_wg_id(); \
        my_reduce[lid] = myval; \
        barrier(); \
        const uint WGS = wg_size; \
        for(uint i=WGS / 2;i>0; i>>= 1) { \
            if(lid < i) { \
                my_reduce[lid] = reduce_op(my_reduce[lid],my_reduce[lid+i]); \
            } \
            barrier(); \
        } \
        myval = my_reduce[0]; \
    } while(false)

#define REDUCE_OP_ADD(x,y) ((x) + (y))
#define REDUCE_OP_MAX(x,y) max((x),(y))

#define my_work_group_reduce_add(val, wg_size) REDUCE_USING_OP(val,REDUCE_OP_ADD, wg_size)
#define my_work_group_reduce_max(val, wg_size) REDUCE_USING_OP(val,REDUCE_OP_MAX, wg_size)
