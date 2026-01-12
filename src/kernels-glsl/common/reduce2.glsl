///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////

#ifndef WG_SIZE
	#define WG_SIZE 256
#endif

#ifndef my_get_local_wg_id
#define my_get_local_wg_id() ((get_local_id(2) * get_local_size(1) * get_local_size(0)) + (get_local_id(1) * get_local_size(0)) + get_local_id(0))
#endif

#define REDUCE_PREPARE_X2(wg_size, dtype) shared dtype my_reduce_x2[2][wg_size]
#define REDUCE_USING_OP_X2(myval,reduce_op, wg_size) \
    do { \
        uint lid = my_get_local_wg_id(); \
        my_reduce_x2[0][lid] = myval[0]; \
        my_reduce_x2[1][lid] = myval[1]; \
        barrier(); \
        const uint WGS = wg_size; \
        for(uint i=WGS / 2;i>0; i>>= 1) { \
            if(lid < i) { \
                my_reduce_x2[0][lid] = reduce_op(my_reduce_x2[0][lid],my_reduce_x2[0][lid+i]); \
                my_reduce_x2[1][lid] = reduce_op(my_reduce_x2[1][lid],my_reduce_x2[1][lid+i]); \
            } \
            barrier(); \
        } \
        myval[0] = my_reduce_x2[0][0]; \
        myval[1] = my_reduce_x2[1][0]; \
    } while(false)

#define REDUCE_X2_OP_ADD(x,y) ((x) + (y))
#define REDUCE_X2_OP_MAX(x,y) max((x),(y))

#define my_work_group_reduce_add_x2(val, wg_size) REDUCE_USING_OP_X2(val,REDUCE_X2_OP_ADD, wg_size)
#define my_work_group_reduce_max_x2(val, wg_size) REDUCE_USING_OP_X2(val,REDUCE_X2_OP_MAX, wg_size)

