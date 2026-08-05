#include "../common/defs.glsl"
#define USE_SPEC_CONSTANTS 1
#include "../common/reduce2.glsl"

#ifndef SECOND_REDUCE_SIZE
#define SECOND_REDUCE_SIZE 1
#endif

#ifndef BACKWARD
#define BACKWARD 1
#endif

layout(constant_id = 0) const uint WG_SIZE = 256;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer x_buf { float x[]; };
	#if BACKWARD == 1
		layout(binding = 1, std430) readonly buffer dy_buf { float dy[]; };
		layout(binding = 2, std430) buffer dyx_sum_buf { float dyx_sum[]; };
		layout(binding = 3, std430) buffer dy_sum_buf { float dy_sum[]; };
	#else
		#if SECOND_REDUCE_SIZE == 1
			layout(binding = 1, std430) buffer x_mean_buf { float x_mean[]; };
			layout(binding = 2, std430) buffer x_var_buf { float x_var[]; };
		#else
			layout(binding = 1, std430) buffer x_sum_buf { float x_sum[]; };
			layout(binding = 2, std430) buffer x2_sum_buf { float x2_sum[]; };
		#endif
	#endif
#endif

REDUCE_PREPARE_X2(WG_SIZE,float);

layout(push_constant, std430) uniform compute
{
	int batch; int channels; int HW;
	#if USE_BDA
		__global float const *x,
	#endif
	uint x_offset;
#if BACKWARD == 1
	#if USE_BDA
		__global float const *dy;
	#endif
	uint dy_offset;
	#if USE_BDA
		__global float *dyx_sum;
	#endif
	uint dyx_sum_offset;
	#if USE_BDA
		__global float *dy_sum;
	#endif
	uint dy_sum_offset;
#else
	#if SECOND_REDUCE_SIZE == 1
		#if USE_BDA
			__global float *x_mean;
		#endif
		uint x_mean_offset;
		#if USE_BDA
			__global float *x_var;
		#endif
		uint x_var_offset;
	#else
		#if USE_BDA
			__global float *x_sum;
		#endif
		uint x_sum_offset;
		#if USE_BDA
			__global float *x2_sum;
		#endif
		uint x2_sum_offset;
	#endif          
#endif                
};
          
void main()
{
    uint feature = get_global_id(1);
    if(feature >= channels)
        return;

    // x   += x_offset;
#if BACKWARD == 1
    // dy   += dy_offset;
    // dy_sum += dy_sum_offset;
    // dyx_sum += dyx_sum_offset;
#else
  #if SECOND_REDUCE_SIZE == 1
    //x_mean += x_mean_offset;
    //x_var  += x_var_offset;
  #else
    //x_sum += x_sum_offset;
    //x2_sum += x2_sum_offset;
  #endif
#endif    
    
    // x  += feature * HW;
#if BACKWARD == 1
    // dy += feature * HW;
#endif    

    uint items = batch * HW;
    const uint wg_size2 = WG_SIZE * SECOND_REDUCE_SIZE;
    uint items_per_wg = (items + wg_size2 - 1) / wg_size2;
    uint my_start = items_per_wg * get_global_id(0); // it is same as local id for 1stage reduce
    uint my_end   = min(my_start + items_per_wg,items);

    float sum1 = 0;
    float sum2 = 0;
    uint b  = my_start / HW;
    uint rc = my_start % HW;

	// why 16? what is this magic number? we shall find out later c:
    // #pragma unroll(16)
    for(uint index = my_start; index <my_end; index ++)
    {
        if(b < batch && rc < HW)
        {
            uint pos = b*(channels * HW) + rc;
            #if BACKWARD == 1
                float xv  =   x[pos + x_offset + (feature * HW)];
                float dyv =  dy[pos + dy_offset + (feature * HW)];
                sum1 += xv*dyv;
                sum2 += dyv;
            #else
                float val =   x[pos + x_offset + feature * HW];
                sum1 += val;
                sum2 += val*val;
            #endif
        }
        rc++;
        if(rc == HW)
        {
            rc = 0;
            b ++;
        }
    }

	// moved outside main function body
    // REDUCE_PREPARE_X2(WG_SIZE,float);

    vec2 sums = vec2(sum1, sum2);

    my_work_group_reduce_add_x2(sums, WG_SIZE);
    sum1 = sums[0];
    sum2 = sums[1];

    if(get_local_id(0) == 0)
    {
        #if SECOND_REDUCE_SIZE == 1
            uint pos = feature;
        #else
            uint pos = feature + channels * get_group_id(0);
        #endif
        #if BACKWARD == 1
			dyx_sum[pos + dyx_sum_offset] = sum1;
            dy_sum[pos + dy_sum_offset] = sum2;
        #else
            #if SECOND_REDUCE_SIZE == 1
				float mean_val  = sum1 / (batch * HW);
				float mean2_val = sum2 / (batch * HW);
				x_mean[pos + x_mean_offset] = mean_val;
				x_var[pos + x_var_offset] = mean2_val - mean_val*mean_val;
            #else
				x_sum[pos + x_sum_offset] = sum1;
				x2_sum[pos + x2_sum_offset] = sum2;
            #endif
        #endif
    }
}
