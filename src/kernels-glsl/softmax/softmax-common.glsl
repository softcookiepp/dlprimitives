#include "../common/defs.glsl"
#include "../common/reduce.glsl"


#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#ifndef ITEMS_PER_WI
#define ITEMS_PER_WI 1
#endif

#ifndef LOG_SM
#define LOG_SM 0
#endif

#ifndef CALC_LOSS
#define CALC_LOSS 0
#endif

#if CALC_LOSS==1
#include "../common/atomic.glsl"
#endif

#define LOCAL_ITEMS_LIMIT 32
