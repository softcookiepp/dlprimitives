#version 450
#include "../common/defs.glsl"

#ifndef TILE_SIZE_M
#define TILE_SIZE_M 128
#endif
#ifndef TILE_SIZE_N
#define TILE_SIZE_N 128
#endif
#ifndef BLOCK_SIZE_N
#define BLOCK_SIZE_N 8
#endif
#ifndef BLOCK_SIZE_M
#define BLOCK_SIZE_M 8
#endif

#ifndef TILE_SIZE_K
#define TILE_SIZE_K 16
#endif 

#ifndef TILE_OFFSET
#define TILE_OFFSET 1
#endif

#ifndef ZORDER
#define ZORDER 0
#endif


#ifndef ATRANS
#define ATRANS 0
#endif

#ifndef BTRANS
#define BTRANS 0
#endif

#ifndef BIAS
#define BIAS 0
#endif

#ifndef GROUPS
#define GROUPS 1
#endif

#define BLOCK_SIZE_NY (BLOCK_SIZE_N*BLOCK_SIZE_M)
#define BLOCKS_IN_TILE_N (TILE_SIZE_N / BLOCK_SIZE_N)
#define BLOCKS_IN_TILE_M (TILE_SIZE_M / BLOCK_SIZE_M)
#define WG_SIZE (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)

#if 0
	#define ALIGN_FLOAT4
#else
	#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))
#endif

#ifndef CONVGEMM
#define CONVGEMM 0
#endif

#if CONVGEMM == 3 || REDUCE_K > 1
	#include "../common/atomic.glsl"
	#if ACTIVATION != ACTIVATION_IDENTITY
		# error "Can't use activation with atomic ops"
	#endif
#endif

#if CONVGEMM != 0
	#include "get_img_value.glsl"
#endif



#if CONVGEMM == 0 || CONVGEMM == 3
#  if BTRANS == 0
#    define get_B(r,c) (B_arr[(r)*ldb + (c) + B])
#  else
#    define get_B(r,c) (B_arr[(c)*ldb + (r) + B])
#  endif
#else
#  if BTRANS == 0
#    define get_B(r,c) get_img_value(B,r,c)
#  else
#    define get_B(r,c) get_img_value(B,c,r)
#  endif
#endif

#if CONVGEMM  == 0 || CONVGEMM == 1
    #if  ATRANS == 0
        #define get_A(r,c) (A_arr[(r)*lda + (c) + A])
    #else
        #define get_A(r,c) (A_arr[(c)*lda + (r) + A])
    #endif
#else
    float get_y_value(uint row,uint matrix_col, uint A_addr,uint ldc,uint M)
    {
        uint batch = matrix_col / IM2COL_OCHAN;
        uint incol = matrix_col % IM2COL_OCHAN;
        uint offset = batch * (IM2COL_OCHAN * GROUPS) * M + incol;
        uint index =row*IM2COL_OCHAN + offset;
        return A_arr[index + A_addr];
    }

    #if CONVGEMM == 3
        #define GET_Y_STEP K_src
    #else
        #define GET_Y_STEP M
    #endif
    #if  ATRANS == 0
        #define get_A(r,c) (get_y_value(r,c,A,lda,GET_Y_STEP))
    #else
        #define get_A(r,c) (get_y_value(c,r,A,lda,GET_Y_STEP))
    #endif

#endif

#define lA(x,y) a_tile[(x)][(y) / BLOCK_SIZE_M][(y) % BLOCK_SIZE_M]
#define lB(x,y) b_tile[(x)][(y) / BLOCK_SIZE_N][(y) % BLOCK_SIZE_N]

#define lA_coords(x, y) uvec3((x), (y) / BLOCK_SIZE_M, (y) % BLOCK_SIZE_M)
#define lB_coords(x, y) uvec3((x), (y) / BLOCK_SIZE_N, (y) % BLOCK_SIZE_N)

#define lA_load_store_from_vec(pos) a_tile[(pos).x][(pos).y ][(pos).z]
#define lB_load_store_from_vec(pos) b_tile[(pos).x][(pos).y ][(pos).z]

#define lA_load_store(x, y) a_tile[(x)][(y) / BLOCK_SIZE_M][(y) % BLOCK_SIZE_M]
#define lB_load_store(x, y) b_tile[(x)][(y) / BLOCK_SIZE_N][(y) % BLOCK_SIZE_N]


#if TILE_SIZE_M != TILE_SIZE_N
#error "Unsupported condif"
#endif

#if defined(cl_intel_subgroups)
#define INTEL_PLATFORM 1
#else
#define INTEL_PLATFORM 0
#endif

#define vload1(off,addr) ((addr)[off])
#define vstore1(val,off,addr) ((addr)[off]=(val))

#if BLOCK_SIZE_M == 1
#define vloadM vload1
#define vstoreM vstore1
#define floatM float
#elif BLOCK_SIZE_M == 4
#define vloadM vload4
#define vstoreM vstore4
#define floatM float4
#elif BLOCK_SIZE_M == 8
#define vloadM vload8
#define vstoreM vstore8
#define floatM float8
#elif BLOCK_SIZE_M == 16 
#define vloadM vload16
#define vstoreM vstore16
#define floatM float16
#endif

#if BLOCK_SIZE_N == 1
#define vloadN vload1
#define vstoreN vstore1
#define floatN float
#elif BLOCK_SIZE_N == 4
#define vloadN vload4
#define vstoreN vstore4
#define floatN float4
#elif BLOCK_SIZE_N == 8
#define vloadN vload8
#define vstoreN vstore8
#define floatN float8
#elif BLOCK_SIZE_N == 16 
#define vloadN vload16
#define vstoreN vstore16
#define floatN float16
#endif

#if TILE_SIZE_K == 1
#define vloadK vload1
#define vstoreK vstore1
#define floatK float
#elif TILE_SIZE_K == 4
#define vloadK vload4
#define vstoreK vstore4
#define floatK float4
#elif TILE_SIZE_K == 8
#define floatK float8
#define vloadK vload8
#define vstoreK vstore8
#elif TILE_SIZE_K == 16 
#define vloadK vload16
#define vstoreK vstore16
#define floatK float16
#endif


#ifndef BATCH_GEMM
#define BATCH_GEMM 0
#endif

#ifndef REDUCE_K
#define REDUCE_K 1
#endif

#if GROUPS == 1 && REDUCE_K == 1 && BATCH_GEMM == 0
#define DIM_M 0
#define DIM_N 1
#define DIM_G 2
#define EXTRA_DIM 0
#else
#define DIM_M 1
#define DIM_N 2
#define DIM_G 0
#define EXTRA_DIM 1
#endif

uint zorder_a(uint x)
{
    return
          ((x & (1<<0)) >> 0 )
        | ((x & (1<<2)) >> 1 )
        | ((x & (1<<4)) >> 2 )
        | ((x & (1<<6)) >> 3 )
        | ((x & (1<<8)) >> 4 )
        | ((x & (1<<10)) >> 5 )
        | ((x & (1<<12)) >> 6 )
        | ((x & (1<<14)) >> 7 )
        | ((x & (1<<16)) >> 8 )
        | ((x & (1<<18)) >> 9 )
        | ((x & (1<<20)) >> 10 )
        | ((x & (1<<22)) >> 11 )
        | ((x & (1<<24)) >> 12 );
}
uint zorder_b(uint x)
{
    return zorder_a(x>>1);
}

#define local_wg_size (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)
#define load_step (TILE_SIZE_M * TILE_SIZE_K / local_wg_size)

#if INTEL_PLATFORM == 0
	shared float a_tile[TILE_SIZE_K][BLOCKS_IN_TILE_M][BLOCK_SIZE_M+TILE_OFFSET];
    shared float b_tile[TILE_SIZE_K][BLOCKS_IN_TILE_N][BLOCK_SIZE_N+TILE_OFFSET];
	
	shared uvec3 aP[load_step];
    shared uvec3 bP[load_step];
#endif


#if INTEL_PLATFORM == 1
// no idea how to do this with Vulkan
__attribute__((intel_reqd_sub_group_size(8)))
#endif
#if EXTRA_DIM == 0
layout(local_size_x = BLOCKS_IN_TILE_M, local_size_y = BLOCKS_IN_TILE_N, local_size_z = 1) in;
#else
layout(local_size_x = 1, local_size_y = BLOCKS_IN_TILE_M, local_size_z = BLOCKS_IN_TILE_N) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer A_buf { float A_arr[]; };
	layout(binding = 1, std430) readonly buffer B_buf { float B_arr[]; };
	layout(binding = 2, std430) buffer C_buf { float C_arr[]; };
	#if BIAS != 0
		layout(binding = 3, std430) readonly buffer bias_buf { float bias_arr[]; };
	#endif
#endif

layout(push_constant, std430) uniform sgemm
{
#if BATCH_GEMM == 1
	uint batches;
#endif        
	uint M;
	uint N;
	uint K;
	#if USE_BDA
		__global const float * restrict A;
	#endif
	uint offset_A;
#if BATCH_GEMM == 1
	uint batch_stride_a;
#endif          
	uint lda;
	#if USE_BDA
		__global const float * restrict B;
	#endif
	uint offset_B;
#if BATCH_GEMM == 1
        uint batch_stride_b;
#endif                
	uint ldb;
	#if USE_BDA
		__global float * restrict C;
	#endif
	uint offset_C;
#if BATCH_GEMM == 1
	uint batch_stride_c;
#endif                
	uint ldc;
	float beta_factor;
#if BIAS != 0
	#if USE_BDA
		__global const float * restrict bias;
	#endif
	uint offset_bias;
#endif
};
        
void main()
{
    uint A = offset_A;
    uint B = offset_B;
    uint C = offset_C;
#if BATCH_GEMM == 1 
    uint batch_id = get_global_id(DIM_G);
    if(batch_id >= batches)
        return;
    A += batch_stride_a * batch_id;
    B += batch_stride_b * batch_id;
    C += batch_stride_c * batch_id;
#endif    

#if CONVGEMM > 0 && GROUPS > 1
    if(get_global_id(DIM_G) >= REDUCE_K * GROUPS)
        return;
    uint group = get_global_id(DIM_G) / REDUCE_K;
    #if CONVGEMM == 1
        A += M*K*group;
        B += SRC_COLS*SRC_ROWS*CHANNELS_IN*group;
        C += (IM2COL_OCHAN) * M *group;
        #if BIAS != 0
        bias += M*group;
        #endif
    #elif CONVGEMM == 2
        // M = channels_out / groups
        uint step_g_y = (M*IM2COL_OCHAN);
        uint step_g_x = (SRC_COLS*SRC_ROWS) * CHANNELS_IN;
        uint step_g_w = M*(CHANNELS_IN*KERN_W*KERN_H);
        
        A += step_g_y * group;
        B += step_g_x * group;
        C += step_g_w * group;
    #elif CONVGEMM == 3
        // K = channels_out / group
        uint step_g_y = (K*IM2COL_OCHAN);
        uint step_g_x = (SRC_COLS*SRC_ROWS) * CHANNELS_IN;
        uint step_g_w = K*(CHANNELS_IN*KERN_W*KERN_H);
        
        A += step_g_y * group;
        B += step_g_w * group;
        C += step_g_x * group;
    #else
    #error "Invalid CONVGEMM Value"
    #endif
#endif   

#if ZORDER == 1
    uint gr_m = get_group_id(DIM_M);
    uint gr_n = get_group_id(DIM_N);
    uint gr_size_m = get_num_groups(DIM_M);
    uint gr_size_n = get_num_groups(DIM_N);
    if(gr_size_m == gr_size_n && popcount(gr_size_m) == 1) {
        uint grs  = gr_n * gr_size_m + gr_m;
        gr_n = zorder_a(grs);
        gr_m = zorder_b(grs);
    }
#else
    uint gr_m = get_group_id(DIM_M);
    uint gr_n = get_group_id(DIM_N);
#endif
    uint tile_row0 = gr_m*TILE_SIZE_M;
    uint tile_col0 = gr_n*TILE_SIZE_N;

#if ZORDER == 1
    if(tile_row0 >= M || tile_col0 >= N)
        return;
#endif        

    uint row = tile_row0 + get_local_id(DIM_M) * BLOCK_SIZE_M;
    uint col = tile_col0 + get_local_id(DIM_N) * BLOCK_SIZE_N;


    uint lid0 = get_local_id(DIM_M);
    uint lid1 = get_local_id(DIM_N);
    
    uint local_tile_id = lid0 * get_local_size(DIM_N) + lid1;

    //#define local_wg_size (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)
    //#define load_step (TILE_SIZE_M * TILE_SIZE_K / local_wg_size)

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N];
    for (uint i = 0; i < BLOCK_SIZE_M; i += 1)
    {
		for (uint j = 0; j < BLOCK_SIZE_N; j += 1)
		{
			c[i][j] = 0.0;
		}
	}
    
    uint K_src = K;

#if INTEL_PLATFORM == 0
    float ap[BLOCK_SIZE_M];
    float bp[BLOCK_SIZE_N];

#else
    #if ATRANS == 1
    float a[TILE_SIZE_K][BLOCK_SIZE_M];
    #define pA(ind1,ind2) (a[(ind2)][(ind1)])
    #else
    float a[BLOCK_SIZE_M][TILE_SIZE_K];
    #define pA(ind1,ind2) (a[(ind1)][(ind2)])
    #endif
#endif    

#if REDUCE_K > 1
    uint KS = (K + REDUCE_K - 1) / REDUCE_K;
    uint sec = get_global_id(DIM_G) % REDUCE_K;
    uint k_start=KS * sec;
    K = min(K_src,KS * (sec + 1));
    uint k = k_start;
#else
    uint k=0;
#endif

#if TILE_SIZE_N == TILE_SIZE_M && TILE_SIZE_K % load_step  == 0 && load_step <= TILE_SIZE_K
#define LOAD_VARIANT 0
#else
#define LOAD_VARIANT 1
#endif

#if INTEL_PLATFORM == 0

    #if LOAD_VARIANT == 1
    uint dM[load_step];
    uint dN[load_step];
    uint dK [load_step];
	
	UNROLL(load_step)
    for(uint i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE)
    {
        uint tile_kdir = read_pos / TILE_SIZE_M;
        uint tile_tdir = read_pos % TILE_SIZE_M;
        dM[i] = tile_tdir + tile_row0;
        dN[i] = tile_tdir + tile_col0;
        dK[i]  = tile_kdir;
        aP[i] = lA_coords(tile_kdir,tile_tdir);
        bP[i] = lB_coords(tile_kdir,tile_tdir);
    }
    barrier();
    #endif


 
    for(;k<K;k+=TILE_SIZE_K)
    {
        #if LOAD_VARIANT == 0
        {
            uint tile_kdir0 = local_tile_id / TILE_SIZE_M;
            uint tile_tdir  = local_tile_id % TILE_SIZE_M;
            uint a_row = tile_tdir + tile_row0;
            uint b_col = tile_tdir + tile_col0;

            if(a_row >= M)
            {
                UNROLL(load_step)
                for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lA_load_store(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_M)) {
                    UNROLL(load_step)
                    for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        uint k_rc  = tile_kdir + k;
                        lA_load_store(tile_kdir,tile_tdir) = get_A(a_row,k_rc);
                    }
                }
                else {
                    UNROLL(load_step)
                    for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        uint k_rc  = tile_kdir + k;
                        lA_load_store(tile_kdir,tile_tdir) = k_rc < K ? get_A(a_row,k_rc) : 0.0f;
                    }
                }
            }
            if(b_col >= N) {
                UNROLL(load_step)
                for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lB_load_store(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_N)) {
                    UNROLL(load_step)
                    for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        uint k_rc  = tile_kdir + k;
                        lB_load_store(tile_kdir,tile_tdir) = get_B(k_rc,b_col);
                    }
                }
                else {
                    UNROLL(load_step)
                    for(uint i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        uint k_rc  = tile_kdir + k;
                        lB_load_store(tile_kdir,tile_tdir) = k_rc < K ? get_B(k_rc,b_col) : 0.0f;
                    }
                }
            }

            barrier();
        }
        #else
        {
            if(tile_row0 + TILE_SIZE_M <= M && k + TILE_SIZE_K <= K) {
                UNROLL(load_step)
                for(uint i=0;i<load_step;i++) {
                    uint a_row = dM[i];
                    uint k_rc  = dK[i] + k;
                    //*aP[i] =  get_A(a_row,k_rc);
                    lA_load_store_from_vec(aP[i]) = get_A(a_row,k_rc);
                }
            }
            else {
                UNROLL(load_step)
                for(uint i=0;i<load_step;i++) {
                    uint a_row = dM[i];
                    uint k_rc  = dK[i] + k;
                    //*aP[i] = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
                    lA_load_store_from_vec(aP[i]) = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
                }
            }
            if(tile_col0 + TILE_SIZE_N <= N && k + TILE_SIZE_K <= K) {
                UNROLL(load_step)
                for(uint i=0;i<load_step;i++) {
                    uint k_rc  = dK[i]  + k;
                    uint b_col = dN[i];
                    //*bP[i] = get_B(k_rc,b_col);
                    lB_load_store_from_vec(bP[i]) = get_B(k_rc,b_col);
                }
            }
            else {
                UNROLL(load_step)
                for(uint i=0;i<load_step;i++) {
                    uint k_rc  = dK[i]  + k;
                    uint b_col = dN[i];
                    //*bP[i] = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;
					lB_load_store_from_vec(bP[i]) = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;
                }
            }
            barrier();
        }
        #endif

        // Mutliplication loop
        UNROLL(4)
        for(uint dk=0;dk<TILE_SIZE_K;dk++) {
            UNROLL(BLOCK_SIZE_M)
            for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                ap[dr] = a_tile[dk][lid0][dr];
            }
            UNROLL(BLOCK_SIZE_N)
            for(uint dc=0;dc<BLOCK_SIZE_N;dc++) {
                bp[dc] = b_tile[dk][lid1][dc];
            }
            UNROLL(BLOCK_SIZE_M)
            for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                UNROLL(BLOCK_SIZE_N)
                for(uint dc=0;dc<BLOCK_SIZE_N;dc++) {
                    c[dr][dc] = fma(ap[dr],bp[dc],c[dr][dc]);
                }
            }
        }

        barrier();
    }
#else // INTEL_PLATFORM == 1
    // for intel we don't use local memory
    // we use optimized loads from global memory for A
    // and intel_sub_group_shuffle for optimal loading B
    for(;k<K;k+=TILE_SIZE_K) {
        if(row + BLOCK_SIZE_M - 1 < M && k + TILE_SIZE_K-1 < K) {
            #if CONVGEMM == 2 || CONVGEMM == 3
                UNROLL(BLOCK_SIZE_M)
                for(uint dr=0;dr<BLOCK_SIZE_M;dr++){
                    for(uint dk=0;dk < TILE_SIZE_K;dk++) {
                        pA(dr,dk)=get_A(row+dr,k+dk);
                    }
                }
            #else
                #if ATRANS == 0
                    UNROLL(BLOCK_SIZE_M)
                    for(uint dr=0;dr<BLOCK_SIZE_M;dr++){
                        floatK v=vloadK(0,&get_A(row+dr,k));
                        vstoreK(v,0,a[dr]);
                    }
                #else // ATRANS
                    UNROLL(TILE_SIZE_K)
                    for(uint dk=0;dk<TILE_SIZE_K;dk++){
                        floatM v=vloadM(0,&get_A(row,k+dk));
                        vstoreM(v,0,a[dk]);
                    }
                #endif
            #endif
        }
        else {
            UNROLL(BLOCK_SIZE_M)
            for(uint dr=0;dr<BLOCK_SIZE_M;dr++){
                UNROLL(TILE_SIZE_K)
                for(uint dk=0;dk < TILE_SIZE_K;dk++) {
                    pA(dr,dk) = (row + dr < M && k+dk < K) ? get_A(row+dr,k+dk): 0;
                }
            }
        }

        UNROLL(TILE_SIZE_K)
        for(uint dk=0;dk<TILE_SIZE_K;dk++) {
            if(k + dk >= K)
                continue;
            #if BLOCK_SIZE_N == 8
                uint mycol = col + get_sub_group_local_id();
                float myv = (mycol < N) ? get_B(k+dk,col + get_sub_group_local_id()) : 0;
                UNROLL(BLOCK_SIZE_N)
                for(uint dc=0;dc<BLOCK_SIZE_N;dc++){
                    float b_dc = uintel_sub_group_shuffle(myv,dc);
                    for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                        c[dr][dc] = fma(pA(dr,dk),b_dc,c[dr][dc]);
                    }
                }
            #else
                UNROLL(BLOCK_SIZE_N)
                for(uint dc=0;dc<BLOCK_SIZE_N;dc++){
                    float b_dc = (col + dc < N) ? get_B(k+dk,col+dc) : 0;
                    for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                        c[dr][dc] = fma(pA(dr,dk),b_dc,c[dr][dc]);
                    }
                }
            #endif
        }
    }
#endif // INTEL_PLATFORM = 1


#if BIAS != 0
    bias += offset_bias;
#endif

#if BIAS == 1
    #if REDUCE_K > 1
    if(k_start == 0)
    #endif
    {
        float offset;
        UNROLL(BLOCK_SIZE_M)
        for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
            offset = row + dr < M ? bias[(row+dr)] : 0.0f;
            UNROLL()
            for(uint dc=0;dc<BLOCK_SIZE_N;dc++) {
                c[dr][dc] += offset;
            }
        }
    }
#elif BIAS == 2
    #if REDUCE_K > 1
    if(k_start == 0)
    #endif
    {
        float offset;
        UNROLL(BLOCK_SIZE_N)
        for(uint dc=0;dc<BLOCK_SIZE_N;dc++) {
            offset = (col + dc) < N ? bias[(col+dc)] : 0.0f;
            UNROLL()
            for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                c[dr][dc] += offset;
            }
        }
    }
#endif    

#if CONVGEMM == 1
    {
        UNROLL(BLOCK_SIZE_N)
        for(uint dc=0; dc < BLOCK_SIZE_N; dc++) {
            if(col + dc >= N)
                continue;
            uint matrix_col = col + dc;
            uint batch = matrix_col / IM2COL_OCHAN;
            uint incol = matrix_col % IM2COL_OCHAN;
            uint offset = batch * (IM2COL_OCHAN * GROUPS) * M + incol;
            UNROLL(BLOCK_SIZE_M)
            for(uint dr=0;dr<BLOCK_SIZE_M;dr++) {
                if(row+dr < M) {
                    uint index =(row + dr)*ldc + offset;
                    #if REDUCE_K > 1
                    atomic_addf(C+index,c[dr][dc]);
                    #else
                    if(beta_factor != 0)
                        C_arr[index + C] = fma(C_arr[index + C], beta_factor,ACTIVATION_F(c[dr][dc]));
                    else
                        C_arr[index + C] = ACTIVATION_F(c[dr][dc]);
                    #endif
                }
            }
        }
    }
#else
    {
        UNROLL(LOCK_SIZE_M)
        for (uint dr=0; dr < BLOCK_SIZE_M; dr++)
        {
            UNROLL(BLOCK_SIZE_N)
            for (uint dc=0; dc < BLOCK_SIZE_N; dc++)
            {
                if(row + dr < M && col+dc < N) {
                    #if CONVGEMM == 3
                        add_img_value(C,row+dr,col+dc,c[dr][dc]);
                    #else
                        uint index = (row+dr)*ldc+col+dc;
                        #if REDUCE_K > 1
                        atomic_addf(C+index,ACTIVATION_F(c[dr][dc]));
                        #else
                        if(beta_factor != 0)
                            C_arr[index + C] = fma(C_arr[index + C], beta_factor,ACTIVATION_F(c[dr][dc]));
                        else
                            C_arr[index + C] = ACTIVATION_F(c[dr][dc]);
                        #endif
                    #endif
                }
            }
        }
    }
#endif
}


