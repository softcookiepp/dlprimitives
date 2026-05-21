///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/ops/scal.hpp>
#include <iostream>
#if VULKAN_API
#include <clblast_vk.h>
#endif
#include <dlprim/core/pointwise.hpp>

namespace dlprim {
namespace gpu {
    
    class StandardSGEMMBase  {
    public:
        StandardSGEMMBase(Context &ctx,int M,int N,int K,bool actual_gemm,bool batch_gemm,StandardActivations &activation)
        {
            sep_scale_ = false;
            sep_act_ = false;
            batch_gemm_ = batch_gemm;
            reduce_k_ = 1;
            if(ctx.check_device_extension("cl_intel_subgroups")) {
                block_size_m_ = 8;
                block_size_n_ = 8;
                tile_size_k_ = 4;
                off_ = 0;
                if(M >= 128 && N >= 128) {
                    tile_size_m_ = tile_size_n_ = 128;
                }
                else {
                    tile_size_m_ = tile_size_n_ = 64;
                }
            }
            else {
                if (ctx.is_apple()) {
                    tile_size_m_ = 32;
                    tile_size_n_ = 32;
                    block_size_m_ = 4;
                    block_size_n_ = 4;
                    tile_size_k_ = 16;
                    off_ = 0;
                }
                else if (ctx.is_imagination())
                {
                   tile_size_m_ = 64;
                   tile_size_n_ = 64;
                   block_size_m_ = 8;
                   block_size_n_ = 8;
                   tile_size_k_ = 16;
                   off_ = 1;
                }
                else if(ctx.is_amd() && !actual_gemm) {
                    if(M >= 256 && N >= 256) {
                        tile_size_m_ = 96;
                        tile_size_n_ = 96;
                        block_size_m_ = 6;
                        block_size_n_ = 6;
                        tile_size_k_ = 16;
                        off_ = 0;
                    }
                    else if(M >= 64 && N>= 64) {
                        tile_size_m_ = 64;
                        tile_size_n_ = 64;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 16;
                        off_ = 0;
                    }
                    else if(M >= 32 && N >= 32) {
                        tile_size_m_ = 32;
                        tile_size_n_ = 32;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 32;
                        off_ = 0;
                    }
                    else if(M * N <= 256) {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 1;
                        block_size_n_ = 1;
                        tile_size_k_ = 128;
                        off_ = 0;
                    }
                    else {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 2;
                        block_size_n_ = 2;
                        tile_size_k_ = 64;
                        off_ = 0;
                    }
                }
                else {
                    if(M >= 256 && N >= 256) {
                        tile_size_m_ = 128;
                        tile_size_n_ = 128;
                        block_size_m_ = 8;
                        block_size_n_ = 8;
                        tile_size_k_ = 16;
                        off_ = ctx.is_amd() ? 0 :1;
                    }
                    else if(M >= 128 && N>= 128) {
                        tile_size_m_ = 64;
                        tile_size_n_ = 64;
                        block_size_m_ = 8;
                        block_size_n_ = 8;
                        tile_size_k_ = 16;
                        off_ = ctx.is_amd() ? 0: 1;
                    }
                    else if(M >= 32 && N >= 32) {
                        tile_size_m_ = 32;
                        tile_size_n_ = 32;
                        block_size_m_ = 4;
                        block_size_n_ = 4;
                        tile_size_k_ = 32;
                        off_ = 0;
                    }
                    else if(M * N <= 256) {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 1;
                        block_size_n_ = 1;
                        tile_size_k_ = 128;
                        off_ = 0;
                    }
                    else {
                        tile_size_m_ = 16;
                        tile_size_n_ = 16;
                        block_size_m_ = 2;
                        block_size_n_ = 2;
                        tile_size_k_ = 64;
                        off_ = 0;
                    }
                }
            }
            if(!batch_gemm_) {
                int cores = ctx.estimated_core_count();
                if(cores >= 256 && M * N / (block_size_m_ * block_size_n_) < 4 * cores && K > M*16 && K > N*16) {
                    reduce_k_ = 8;
                    set_scale(ctx,activation);
                }
            }
        }
    protected:
        static int round_up_div(int x,int y)
        {
            return (x + y - 1)/y;
        }
        void check_zorder(Context &ctx,int M,int N)
        {
            ///
            /// on AMD lda % 1024==0 / ldb % 1024==0 wipes cache out - so we reorder 
            //  all ops in Z-order/Morton order
            ///
            zorder_ = 0;
            if(ctx.is_amd() && (M % 1024 == 0 || N % 1024 == 0)) {
                if(M >= N && N*10 >= M)
                    zorder_ = 1;
                if(N >= M && M*10 >= N)
                    zorder_ = 1;
            }
        }
        void calc_dims(int &gs0,int &ls0,int &gs1,int &ls1,int M,int N)
        {
            ls0 = tile_size_m_ / block_size_m_;
            ls1 = tile_size_n_ / block_size_n_; 
            int gr0 = round_up_div(M,tile_size_m_);
            int gr1 = round_up_div(N,tile_size_n_);

            if(zorder_) {
                int gr = std::max(gr0,gr1);
                int n=1;
                while(gr > n) {
                    n<<=1;
                }
                gr0 = gr1 = n;
            }
            
            gs0 = gr0 * ls0;
            gs1 = gr1 * ls1;
        }
        void set_scale(Context &ctx,StandardActivations &activation)
        {
            if(sep_scale_ == false) {
#if VULKAN_API
				tart::program_ptr
#else
                cl::Program const &
#endif
					prog = gpu::Cache::instance().get_program(ctx,"scal");
#if VULKAN_API
                scal_ = prog->getKernel("sscal");
#else
                scal_ = std::move(cl::Kernel(prog,"sscal"));
#endif
                if(activation != StandardActivations::identity) {
#if VULKAN_API
					tart::program_ptr
#else
                    cl::Program const &
#endif
						prog = gpu::Cache::instance().get_program(ctx,"activation",
                                                        "ACTIVATION",int(activation));
#if VULKAN_API
					tart::kernel_ptr k = prog->getKernel("activation");
					act_ = k;
#else
                    cl::Kernel k(prog,"activation");
                    act_ = std::move(k);
#endif
                    activation = StandardActivations::identity;
                    sep_act_ = true;
                }

                sep_scale_ = true;
            }
        }

        void activation(size_t size, 
#if VULKAN_API
			tart::buffer_ptr x,
			size_t x_offset,
#else
			cl::Buffer &x,
			cl_ulong x_offset,
#endif
			ExecutionContext const &ec)
        {
#if VULKAN_API
			act_->setArg(0, uint32_t(size));
			act_->setArg(1, x);
			act_->setArg(2, x_offset);
			act_->setArg(3, x);
			act_->setArg(4, x_offset);
#else
			act_.setArg(0, cl_ulong(size));
			act_.setArg(1, x);
			act_.setArg(2, x_offset);
			act_.setArg(3, x);
			act_.setArg(4, x_offset);
#endif
#if VULKAN_API
			// TODO: ensure the global size for this isn't wrong
			act_->run({size}, {1});
#else
			ec.queue().enqueueNDRangeKernel(act_, cl::NullRange, cl::NDRange(size),cl::NullRange,ec.events(),ec.event("activation"));
#endif
        }
#if VULKAN_API
        void scale(size_t size,float s, const tart::buffer_ptr& x, uint32_t x_offset,ExecutionContext const &ec)
#else
        void scale(size_t size,float s,cl::Buffer &x,cl_ulong x_offset,ExecutionContext const &ec)
#endif
        {
            int wg = 64;
            if(size >= 1024)
                wg = 256;
            int p=0;
#if VULKAN_API 
            scal_->setArg(p++, uint32_t(size));
            scal_->setArg(p++, s);
            scal_->setArg(p++, x);
            scal_->setArg(p++, x_offset);
#else
            scal_.setArg(p++, cl_ulong(size));
            scal_.setArg(p++,s);
            scal_.setArg(p++,x);
            scal_.setArg(p++,x_offset);
#endif
#if VULKAN_API
			std::vector<uint32_t> l({wg});
			std::vector<uint32_t> g = gpu::round_range(size, l);
			g[0] = g[0]/wg;
			scal_->run(g, {});
#else
            cl::NDRange l(wg);
            cl::NDRange g=gpu::round_range(size,l);
            ec.queue().enqueueNDRangeKernel(scal_,cl::NullRange,g,l,ec.events(),ec.event("gemm_beta_scale"));
#endif
        }
        int tile_size_n_,tile_size_m_,tile_size_k_;
        int block_size_n_,block_size_m_;
        int off_;
        int reduce_k_;
        bool sep_scale_;
        bool sep_act_;
        bool batch_gemm_;
#if VULKAN_API
        tart::kernel_ptr scal_;
        tart::kernel_ptr act_;
#else
        cl::Kernel scal_;
        cl::Kernel act_;
#endif
        bool zorder_ = false;
    };

    class StandardSGEMM : public GEMM, public StandardSGEMMBase {
    public:
        StandardSGEMM(  Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int K,
                        int bias,
                        StandardActivations act,
                        int im2col_chan = 0) : 
                StandardSGEMMBase(ctx,M,N,K,true,false,act)
        {
            check_zorder(ctx,M,N);
#if VULKAN_API
            tart::program_ptr prog = Cache::instance().get_program(ctx,"sgemm",
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",bias,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "IM2COL_OCHAN",im2col_chan,
                                        "REDUCE_K",reduce_k_,
                                        "ZORDER",zorder_,
                                        "ACTIVATION",int(act));
            kernel_ = prog->getKernel("sgemm");
#else
            cl::Program const &prog = Cache::instance().get_program(ctx,"sgemm",
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",bias,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "IM2COL_OCHAN",im2col_chan,
                                        "REDUCE_K",reduce_k_,
                                        "ZORDER",zorder_,
                                        "ACTIVATION",int(act));
            kernel_ = cl::Kernel(prog,"sgemm");
#endif
            bias_ = bias;
        }
        static int round_up_div(int x,int y)
        {
            return (x + y - 1)/y;
        }
        virtual void gemm(int M,int N,int K,
#if VULKAN_API
						tart::buffer_ptr &a,
						uint32_t offset_a,
						int lda,
						tart::buffer_ptr &b,
						uint32_t offset_b,
						int ldb,
						tart::buffer_ptr &c,
						uint32_t offset_c,
						int ldc,
						tart::buffer_ptr bias,
						uint32_t bias_offset,
#else
                          cl::Buffer &a,
                          cl_ulong offset_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int ldc,
                          cl::Buffer *bias,
                          cl_ulong bias_offset,
#endif
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {

            ExecutionContext e;
            int kernel_runs = 1 + int(sep_act_) + int(sep_scale_);
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,kernel_runs));
                e=ein.generate_series_context(1,kernel_runs);
                beta = 1.0;
            }
            else {
                e=ein;
            }
            int ind=0;
#if VULKAN_API
            kernel_->setArg(ind++,M);
            kernel_->setArg(ind++,N);
            kernel_->setArg(ind++,K);
            kernel_->setArg(ind++,a);
            kernel_->setArg(ind++,offset_a);
            kernel_->setArg(ind++,lda);
            kernel_->setArg(ind++,b);
            kernel_->setArg(ind++,offset_b);
            kernel_->setArg(ind++,ldb);
            kernel_->setArg(ind++,c);
            kernel_->setArg(ind++,offset_c);
            kernel_->setArg(ind++,ldc);
            kernel_->setArg(ind++,beta);
            if(bias_) {
                DLPRIM_CHECK(bias != nullptr);
                kernel_->setArg(ind++,*bias);
                kernel_->setArg(ind++,bias_offset);
            }
#else
            kernel_.setArg(ind++,M);
            kernel_.setArg(ind++,N);
            kernel_.setArg(ind++,K);
            kernel_.setArg(ind++,a);
            kernel_.setArg(ind++,offset_a);
            kernel_.setArg(ind++,lda);
            kernel_.setArg(ind++,b);
            kernel_.setArg(ind++,offset_b);
            kernel_.setArg(ind++,ldb);
            kernel_.setArg(ind++,c);
            kernel_.setArg(ind++,offset_c);
            kernel_.setArg(ind++,ldc);
            kernel_.setArg(ind++,beta);
            if(bias_) {
                DLPRIM_CHECK(bias != nullptr);
                kernel_.setArg(ind++,*bias);
                kernel_.setArg(ind++,bias_offset);
            }
#endif
            else {
                DLPRIM_CHECK(bias == nullptr);
            }

            int gs0,gs1,ls0,ls1;
            calc_dims(gs0,ls0,gs1,ls1,M,N);
#if VULKAN_API
			std::vector<uint32_t>
#else
            cl::NDRange
#endif
				global,local;
            if(reduce_k_ > 1) {
#if VULKAN_API
				global = {reduce_k_,gs0/ls0,gs1/ls1};
				local = {1,ls0,ls1};
#else
                global = cl::NDRange(reduce_k_,gs0,gs1);
                local =  cl::NDRange(1,ls0,ls1);
#endif
            }
            else {
#if VULKAN_API
				global = {gs0/ls0,gs1/ls1};
                local =  {ls0,ls1};
#else
                global = cl::NDRange(gs0,gs1);
                local =  cl::NDRange(ls0,ls1);
#endif
            }
#if VULKAN_API
			// correct global size
			for (size_t i = 0; i < global.size(); i += 1)
			{
				global[i] = global[i]/local[i];
			}
			// weeeeeeeeeeeeeee
			kernel_->run(global, {});
#else
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event("gemm"));
#endif
            
            if(sep_act_) {
                auto e2 = ein.generate_series_context(kernel_runs-1,kernel_runs);
                activation(size_of_c,c,offset_c,e2);
            }
        }

    private:
#if VULKAN_API
        tart::kernel_ptr kernel_;
#else
        cl::Kernel kernel_;
#endif
        bool bias_;
    };

#if VULKAN_API
	class BlasBatchSGEMM
	{
		tart::device_ptr mDevice = nullptr;
		clblast::Transpose mATrans;
		clblast::Transpose mBTrans;
	public:
		BlasBatchSGEMM(Context &ctx, bool atrans,bool btrans,
			int M,int N,int K, StandardActivations &act)
		{
			mDevice = ctx.device();
			mATrans = atrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
			mBTrans = btrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
		}
		
		virtual void gemm(int batches,int M,int N,int K,
			tart::buffer_ptr a, uint32_t offset_a, int batch_stride_a, int lda,
			tart::buffer_ptr b, uint32_t offset_b, int batch_stride_b, int ldb,
			tart::buffer_ptr c, uint32_t offset_c, int batch_stride_c, int ldc,
			float beta, ExecutionContext const &e)
        {
			//throw std::runtime_error("BlasBatchSGEMM::gemm not implemented");
			
			// adjust matrices based on offset; maybe it will work here?
			size_t byte_offset_a = offset_a*sizeof(float);
			size_t byte_offset_b = offset_b*sizeof(float);
			size_t byte_offset_c = offset_c*sizeof(float);
			if (offset_a > 0)
				a = a->view(byte_offset_a);
			if (offset_b > 0)
				b = b->view(byte_offset_b);
			if (offset_c > 0)
				c = c->view(byte_offset_c);
			if (offset_a | offset_b | offset_c)
				std::cout << "\nWE GOTS AN OFFSET" << std::endl;
			const float alpha = 1.0;
			//if (M == N == K == 1)
			if (false)
			{
				// This is just a batch of 1x1 matrices.
				// As such, it can simply be treated as scalar multiplication.
				
				
				throw std::runtime_error("WHY ARE YOU USING 1???");
			}
			else
			{
				clblast::GemmStridedBatched(clblast::Layout::kRowMajor, mATrans, mBTrans,
					M, N, K, alpha,
					a, 0, lda, batch_stride_a,
					b, 0, ldb, batch_stride_b,
					beta,
					c, 0, ldc, batch_stride_c,
					batches, mDevice);
			}
		}
	};
#endif

    class BatchSGEMM : public StandardSGEMMBase {
    public:
        BatchSGEMM(     Context &ctx,
                        bool atrans,bool btrans,
                        int M,int N,int K,
                        StandardActivations &act
                        ):
                StandardSGEMMBase(ctx,M,N,K,true,true,act)
        {
            DLPRIM_CHECK(act == StandardActivations::identity);
            check_zorder(ctx,M,N);
#if VULKAN_API
			tart::program_ptr
#else
            cl::Program const &
#endif
				prog = Cache::instance().get_program(ctx,"sgemm",
                                        "BATCH_GEMM",1,
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",0,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "IM2COL_OCHAN",0,
                                        "REDUCE_K",0,
                                        "ZORDER",zorder_,
                                        "ACTIVATION",0);
#if VULKAN_API
			throw std::runtime_error("using this right now might be a bad idea");
			kernel_ = prog->getKernel("sgemm");
#else
            kernel_ = cl::Kernel(prog,"sgemm");
#endif
            bias_ = false;
        }
        virtual void gemm(int batches,int M,int N,int K,
#if VULKAN_API
						tart::buffer_ptr a,
						uint32_t offset_a,
						int batch_stride_a,
						int lda,
						tart::buffer_ptr b,
						uint32_t offset_b,
						int batch_stride_b,
						int ldb,
						tart::buffer_ptr c,
						uint32_t offset_c,
#else
                          cl::Buffer &a,
                          cl_ulong offset_a,
                          int batch_stride_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int batch_stride_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
#endif
                          int batch_stride_c,
                          int ldc,
                          float beta,
                          ExecutionContext const &e)
        {

            int ind=0;
#if VULKAN_API
            kernel_->setArg(ind++,batches);
            kernel_->setArg(ind++,M);
            kernel_->setArg(ind++,N);
            kernel_->setArg(ind++,K);
            kernel_->setArg(ind++,a);
            kernel_->setArg(ind++,offset_a);
            kernel_->setArg(ind++,batch_stride_a);
            kernel_->setArg(ind++,lda);
            kernel_->setArg(ind++,b);
            kernel_->setArg(ind++,offset_b);
            kernel_->setArg(ind++,batch_stride_b);
            kernel_->setArg(ind++,ldb);
            kernel_->setArg(ind++,c);
            kernel_->setArg(ind++,offset_c);
            kernel_->setArg(ind++,batch_stride_c);
            kernel_->setArg(ind++,ldc);
            kernel_->setArg(ind++,beta);
#else
            kernel_.setArg(ind++,batches);
            kernel_.setArg(ind++,M);
            kernel_.setArg(ind++,N);
            kernel_.setArg(ind++,K);
            kernel_.setArg(ind++,a);
            kernel_.setArg(ind++,offset_a);
            kernel_.setArg(ind++,batch_stride_a);
            kernel_.setArg(ind++,lda);
            kernel_.setArg(ind++,b);
            kernel_.setArg(ind++,offset_b);
            kernel_.setArg(ind++,batch_stride_b);
            kernel_.setArg(ind++,ldb);
            kernel_.setArg(ind++,c);
            kernel_.setArg(ind++,offset_c);
            kernel_.setArg(ind++,batch_stride_c);
            kernel_.setArg(ind++,ldc);
            kernel_.setArg(ind++,beta);
#endif
           
            int gs0,gs1,ls0,ls1;
            calc_dims(gs0,ls0,gs1,ls1,M,N);
#if VULKAN_API
			std::vector<uint32_t>  local({1,ls0,ls1});
			std::vector<uint32_t> global({batches/local[0],gs0/local[1],gs1/local[2]});
			kernel_->run(global, {});
#else
            cl::NDRange global = cl::NDRange(batches,gs0,gs1);
            cl::NDRange local =  cl::NDRange(1,ls0,ls1);
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event("gemm"));
#endif      
        }

    private:
#if VULKAN_API
		tart::kernel_ptr kernel_;
#else
        cl::Kernel kernel_;
#endif
        bool bias_;
        bool zorder_;
    };
    
#if VULKAN_API
	class BlasConvSGEMM : public GEMM
	{
		tart::device_ptr mDevice;
		clblast::Transpose mATrans;
		clblast::Transpose mBTrans;
		
		GemmOpMode mGemmOpMode;
		size_t mSrcChannels;
		size_t mDstChannels;
		const std::vector<size_t> mKernel;
		const std::vector<size_t> mDilate;
		const std::vector<size_t> mPadding;
		const std::vector<size_t> mStride;
		size_t mGroups;
		
	public:
		BlasConvSGEMM(  Context &ctx,
                    GemmOpMode op_mode,
                    bool atrans,bool btrans,
                    int M,int N,int K,
                    int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
                    int src_channels,int src_rows,int src_cols,
                    int tgt_rows,int tgt_cols,
                    int bias,
                    StandardActivations act,
                    int im2col_chan = 0):
			//StandardSGEMMBase(ctx,M,N,K,false,false,act),
			mGemmOpMode(op_mode),
			mSrcChannels(src_channels),
			mKernel({kernel[0], kernel[1]}),
			mDilate({dilate[0], dilate[1]}),
			mPadding({padding[0], padding[1]}),
			mStride({stride[0], stride[1]}),
			mGroups(groups)
		{
			
			if (mGemmOpMode != GemmOpMode::forward)
				throw std::runtime_error("only forward is implemented right now :c");
			mDevice = ctx.device();
			mATrans = atrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
			mBTrans = btrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
		}
		
		virtual void gemm(int M,int N,int K,
						tart::buffer_ptr &a,
						uint32_t offset_a,
						int lda,
						tart::buffer_ptr &b,
						uint32_t offset_b,
						int ldb,
						tart::buffer_ptr &c,
						uint32_t offset_c,
						int ldc,
						tart::buffer_ptr bias,
						uint32_t bias_offset,
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {
#if 0
			throw std::runtime_error("not implemented");
#else
			clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, mSrcChannels, (size_t)M, (size_t)N,
				size_t(mKernel[0]), size_t(mKernel[1]), size_t(mPadding[0]), size_t(mPadding[1]),
				size_t(mStride[0]), size_t(mStride[1]), size_t(mDilate[0]), size_t(mDilate[1]),
				mDstChannels, //const size_t num_kernels,
				mGroups, // const size_t batch_count,
				a, (size_t)offset_a,
				b, (size_t)offset_b,
				c, (size_t)offset_c,
				mDevice, nullptr);
#endif
		}
		
		
	};
#endif
    class ConvSGEMM : public GEMM, public StandardSGEMMBase {
    public:
        ConvSGEMM(  Context &ctx,
                    GemmOpMode op_mode,
                    bool atrans,bool btrans,
                    int M,int N,int K,
                    int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
                    int src_channels,int src_rows,int src_cols,
                    int tgt_rows,int tgt_cols,
                    int bias,
                    StandardActivations act,
                    int im2col_chan = 0) :
                StandardSGEMMBase(ctx,M,N,K,false,false,act)
        {
#if VULKAN_API
			tart::program_ptr
#else
            cl::Program const &
#endif
            prog = Cache::instance().get_program(ctx,"sgemm",
                                        "TILE_SIZE_M",tile_size_m_,
                                        "TILE_SIZE_N",tile_size_n_,
                                        "BLOCK_SIZE_M",block_size_m_,
                                        "BLOCK_SIZE_N",block_size_n_,
                                        "TILE_SIZE_K",tile_size_k_,
                                        "TILE_OFFSET",off_,
                                        "BIAS",bias,
                                        "ATRANS",int(atrans),
                                        "BTRANS",int(btrans),
                                        "IM2COL_OCHAN",im2col_chan,
                                        "CONVGEMM",int(op_mode),
                                        "KERN_H",  kernel[0], "KERN_W",kernel[1],
                                        "DILATE_H",dilate[0], "DILATE_W",dilate[1],
                                        "PAD_H",   padding[0],"PAD_W",padding[1],
                                        "STRIDE_H",stride[0], "STRIDE_W",stride[1],
                                        "GROUPS",groups,
                                        "CHANNELS_IN",src_channels,
                                        "SRC_COLS",src_cols,
                                        "SRC_ROWS",src_rows,
                                        "IMG_COLS",tgt_cols,
                                        "IMG_ROWS",tgt_rows,
                                        "REDUCE_K",reduce_k_,
                                        "ACTIVATION",int(act));
            if(op_mode == GemmOpMode::backward_data) {
                DLPRIM_CHECK(act == StandardActivations::identity);
                set_scale(ctx,act);
                gemm_name_="conv_gemm_bwd_data";
            }
            else if(op_mode == GemmOpMode::backward_filter)
                gemm_name_="conv_gemm_bwd_filter";
            else
                gemm_name_="conv_gemm";
            kernel_ = 
#if VULKAN_API
				prog->getKernel("sgemm");
#else
				cl::Kernel(prog,"sgemm");
#endif
            bias_ = bias;
            groups_ = groups;
            md_ = int(op_mode);
            k_ = kernel[0];
            pad_ = padding[0];
            s_ = stride[0];
            ci_ = src_channels;
            w_ = src_cols;
        }
        virtual void gemm(int M,int N,int K,
#if VULKAN_API
						tart::buffer_ptr &a,
						uint32_t offset_a,
						int lda,
						tart::buffer_ptr &b,
						uint32_t offset_b,
						int ldb,
						tart::buffer_ptr &c,
						uint32_t offset_c,
						int ldc,
						tart::buffer_ptr bias,
						uint32_t bias_offset,
#else
                          cl::Buffer &a,
                          cl_ulong offset_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
                          int ldc,
                          cl::Buffer *bias,
                          cl_ulong bias_offset,
#endif
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {
            ExecutionContext e;
            int kernel_runs = 1 + int(sep_scale_) + int(sep_act_);
            if(sep_scale_) {
                scale(size_of_c,beta,c,offset_c,ein.generate_series_context(0,kernel_runs));
                e=ein.generate_series_context(1,kernel_runs);
                beta = 1.0;
            }
            else {
                e=ein;
            }
            int ind=0;
#if VULKAN_API
            kernel_->setArg(ind++,M);
            kernel_->setArg(ind++,N);
            kernel_->setArg(ind++,K);
            kernel_->setArg(ind++,a);
            kernel_->setArg(ind++,offset_a);
            kernel_->setArg(ind++,lda);
            kernel_->setArg(ind++,b);
            kernel_->setArg(ind++,offset_b);
            kernel_->setArg(ind++,ldb);
            kernel_->setArg(ind++,c);
            kernel_->setArg(ind++,offset_c);
            kernel_->setArg(ind++,ldc);
            kernel_->setArg(ind++,beta);
            if(bias_) {
                DLPRIM_CHECK(bias != nullptr);
                kernel_->setArg(ind++,*bias);
                kernel_->setArg(ind++,bias_offset);
            }
#else
            kernel_.setArg(ind++,M);
            kernel_.setArg(ind++,N);
            kernel_.setArg(ind++,K);
            kernel_.setArg(ind++,a);
            kernel_.setArg(ind++,offset_a);
            kernel_.setArg(ind++,lda);
            kernel_.setArg(ind++,b);
            kernel_.setArg(ind++,offset_b);
            kernel_.setArg(ind++,ldb);
            kernel_.setArg(ind++,c);
            kernel_.setArg(ind++,offset_c);
            kernel_.setArg(ind++,ldc);
            kernel_.setArg(ind++,beta);
            if(bias_) {
                DLPRIM_CHECK(bias != nullptr);
                kernel_.setArg(ind++,*bias);
                kernel_.setArg(ind++,bias_offset);
            }
#endif
            else {
                DLPRIM_CHECK(bias == nullptr);
            }
           
            int ls0 = tile_size_m_ / block_size_m_;
            int ls1 = tile_size_n_ / block_size_n_; 
            int gs0 = round_up_div(M,tile_size_m_) * tile_size_m_ / block_size_m_;
            int gs1 = round_up_div(N,tile_size_n_) * tile_size_n_ / block_size_n_;
#if VULKAN_API
			std::vector<uint32_t> global,local;
#else
            cl::NDRange global,local;
#endif
            if(groups_ > 1 || reduce_k_ > 1) {
#if VULKAN_API
				local = {1,ls0,ls1};
				global = { (groups_ * reduce_k_)/local[0], gs0/local[1], gs1/local[2]};
#else
                global = cl::NDRange(groups_ * reduce_k_,gs0,gs1);
                local =  cl::NDRange(1,ls0,ls1);
#endif
            }
            else {
#if VULKAN_API
				local = {ls0, ls1, 1};
				global = {gs0/ls0, gs1/ls1, 1};
#else
                global = cl::NDRange(gs0,gs1,1);
                local =  cl::NDRange(ls0,ls1,1);
#endif
            }
#if VULKAN_API
			kernel_->run(global, {});
#else
            e.queue().enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,e.events(),e.event(gemm_name_));
#endif

            if(sep_act_) {
                auto e2 = ein.generate_series_context(kernel_runs-1,kernel_runs);
                activation(size_of_c,c,offset_c,e2);
            }
        }

    private:
        char const *gemm_name_;
#if VULKAN_API
		tart::kernel_ptr kernel_ = nullptr;
		tart::kernel_ptr scal_ = nullptr;
#else
        cl::Kernel kernel_;
        cl::Kernel scal_;
#endif
        bool bias_;
        int groups_;
        int md_;
        int w_;
        int ci_,co_,k_,pad_,s_;
    };

#if VULKAN_API
	// dlprim's existing GEMM kernels are too difficult for me to port to GLSL.
	// So I am simply taking the stuff I developed for CLBlast and putting it here!
	
	class BlasSGEMM : public GEMM
	{
		// taken from clblast
		static bool a_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
		static bool b_want_rotated_(const size_t) { return true; }
		static bool c_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
		
		// Process the user-arguments, computes secondary parameters
		static void ProcessArguments(const clblast::Layout layout, const clblast::Transpose a_transpose, const clblast::Transpose b_transpose,
			const size_t m, const size_t n, const size_t k, size_t& a_one, size_t& a_two,
			size_t& b_one, size_t& b_two, size_t& c_one, size_t& c_two, bool& a_do_transpose,
			bool& b_do_transpose, bool& c_do_transpose, bool& a_conjugate, bool& b_conjugate,
			const size_t gemm_kernel_id)
		{
			// Makes sure all dimensions are larger than zero
			if ((m == 0) || (n == 0) || (k == 0))
			{
				throw std::runtime_error("invalid dimension");
			}

			// Computes whether or not the matrices are transposed in memory. This is based on their layout
			// (row or column-major) and whether or not they are requested to be pre-transposed. Note
			// that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
			// col-major) to be transformed, so transposing requirements are not the same as whether or not
			// the matrix is actually transposed in memory.
			const auto a_rotated = (layout == clblast::Layout::kColMajor && a_transpose != clblast::Transpose::kNo) ||
			(layout == clblast::Layout::kRowMajor && a_transpose == clblast::Transpose::kNo);
			const auto b_rotated = (layout == clblast::Layout::kColMajor && b_transpose != clblast::Transpose::kNo) ||
			(layout == clblast::Layout::kRowMajor && b_transpose == clblast::Transpose::kNo);
			const auto c_rotated = (layout == clblast::Layout::kRowMajor);
			a_do_transpose = a_rotated != a_want_rotated_(gemm_kernel_id);
			b_do_transpose = b_rotated != b_want_rotated_(gemm_kernel_id);
			c_do_transpose = c_rotated != c_want_rotated_(gemm_kernel_id);

			// In case of complex data-types, the transpose can also become a conjugate transpose
			a_conjugate = (a_transpose == clblast::Transpose::kConjugate);
			b_conjugate = (b_transpose == clblast::Transpose::kConjugate);

			// Computes the first and second dimensions of the 3 matrices taking into account whether the
			// matrices are rotated or not
			a_one = (a_rotated) ? k : m;
			a_two = (a_rotated) ? m : k;
			b_one = (b_rotated) ? n : k;
			b_two = (b_rotated) ? k : n;
			c_one = (c_rotated) ? n : m;
			c_two = (c_rotated) ? m : n;
			
		}

#if 0
		void GemmDirect(const size_t m, const size_t n, const size_t k, const float alpha, tart::buffer_ptr& a_buffer,
			const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
			const size_t b_ld, const T beta, tart::buffer_ptr& c_buffer, const size_t c_offset,
			const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
			const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate)
		{
			// Retrieves the proper XgemmDirect kernel from the compiled binary
			const auto name = (a_do_transpose) ? (b_do_transpose ? "xgemm_direct_tt" : "xgemm_direct_tn") : (b_do_transpose ? "xgemm_direct_nt" : "xgemm_direct_nn");
			
			auto prog = Cache::instance().get_program(ctx, "sgemm-from-clblast"
			// TODO: pull all the defs from the CLBlast DB, then throw them in here
			);
			
			auto kernel = prog->getKernel(name);

			// Sets the kernel arguments
			kernel->setArg(0, static_cast<int>(m));
			kernel->setArg(1, static_cast<int>(n));
			kernel->setArg(2, static_cast<int>(k));
			kernel->setArg(3, alpha);
			kernel->setArg(4, beta);
			kernel->setArg(5, a_buffer);
			kernel->setArg(6, static_cast<int>(a_offset));
			kernel->setArg(7, static_cast<int>(a_ld));
			kernel->setArg(8, b_buffer);
			kernel->setArg(9, static_cast<int>(b_offset));
			kernel->setArg(10, static_cast<int>(b_ld));
			kernel->setArg(11, c_buffer);
			kernel->setArg(12, static_cast<int>(c_offset));
			kernel->setArg(13, static_cast<int>(c_ld));
			kernel->setArg(14, static_cast<int>(c_do_transpose));
			kernel->setArg(15, static_cast<int>(a_conjugate));
			kernel->setArg(16, static_cast<int>(b_conjugate));
			
			// provide the same buffers twice to get around GLSL's lack of pointer casting
			kernel->setArg(17, a_buffer());
			kernel->setArg(18, b_buffer());
			
			// default WGD is 8
			const auto WGD = 8;
			
			// default MDIMCD is also 8
			const auto MDIMCD = 8;
			
			// default NDIMCD is, yet again, 8
			const auto NDIMCD = 8;

			// Computes the global and local thread sizes
			const auto m_ceiled = Ceil(m, WGD); // where do we even get this from?
			const auto n_ceiled = Ceil(n, WGD); // or this? or any of the db_ values?
			const auto global =
					std::vector<size_t>{//	CeilDiv(m * db_["MDIMCD"], db_["WGD"]),
															//	CeilDiv(n * db_["NDIMCD"], db_["WGD"])
															(m_ceiled * MDIMCD) / WGD, (n_ceiled * NDIMCD) / WGD};
			const auto local = std::vector<size_t>{MDIMCD, NDIMCD};

			// Launches the kernel
			kernel->run(global, local);
		}
#endif
		
    public:
        BlasSGEMM(Context &ctx, bool atrans,bool btrans, int M,int N,int K, int bias, StandardActivations act, int im2col_chan = 0) :
			mUseBias(bias)
        {
			if (act != StandardActivations::identity)
				throw std::runtime_error("non-identity activations not implemented yet");
			mDevice = ctx.device();
			mATrans = atrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
			mBTrans = btrans ? clblast::Transpose::kYes : clblast::Transpose::kNo;
        }
        
        virtual void gemm(int M,int N,int K,
						tart::buffer_ptr &a,
						uint32_t offset_a,
						int lda,
						tart::buffer_ptr &b,
						uint32_t offset_b,
						int ldb,
						tart::buffer_ptr &c,
						uint32_t offset_c,
						int ldc,
						tart::buffer_ptr bias,
						uint32_t bias_offset,
                          float beta,
                          int size_of_c,
                          ExecutionContext const &ein)
        {
#if 1
			if (mUseBias)
			{
				// C should be row-major, width of N, height of M
				// bias seems to be assumed to be contiguous, aside from the offset.
				// Which means in-place biasing should be easy to implement
				throw std::runtime_error("not implemented!");
				for (size_t i = 0; i < M; i += 1)
				{
					uint32_t c_row_offset = (ldc*i) + offset_c;
					// pretty sure x_inc is just 1, unless C somehow has strides.
					clblast::Copy<float>(N, bias, bias_offset, 1, c, c_row_offset, 1, mDevice);
				}
			}
			
			const float alpha = 1.0;
			clblast::Gemm(clblast::Layout::kRowMajor, mATrans, mBTrans, M, N, K, alpha,
				a, offset_a, lda, b, offset_b, ldb, beta, c, offset_c, ldc, mDevice);
			
			
#else
			auto layout = clblast::Layout::kRowMajor
			auto aTrans = mATrans;
			auto bTrans = mBTrans;
			float alpha = 1.0;
			
			size_t a_one;
			size_t a_two;
			size_t b_one;
			size_t b_two;
			size_t c_one;
			size_t c_two;
			
			bool a_do_transpose;
			bool b_do_transpose;
			bool c_do_transpose;
			
			bool a_conjugate;
			bool b_conjugate;
			
			ProcessArguments(layout, mATrans, mBTrans,
				M, N, K, a_one, a_two, b_one, b_two, c_one, c_two, a_do_transpose,
				b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, 0);
			
			// just skip testing for now
			GemmDirect(M, N, K, alpha, a, offset_a, lda, b, offset_b, ldb, beta, c, offset_c, ldc,
				 a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate);
			
#endif
        }

    private:
		// here we store program, since there are multiple kernels
		tart::program_ptr mProgram = nullptr;
		tart::program_ptr mBiasProgram = nullptr;
		tart::device_ptr mDevice = nullptr;
		
		clblast::Transpose mATrans;
		clblast::Transpose mBTrans;
		
		const bool mSepAct = true;
		const bool mUseBias;
    };
	
#endif

    std::unique_ptr<GEMM> GEMM::get_optimal_gemm(
            Context &ctx,DataType dtype,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int bias,
            StandardActivations act,
            int im2col_chan)
    {
#if VULKAN_API
		DLPRIM_CHECK(dtype == float_data);
		std::unique_ptr<GEMM> g = std::make_unique<BlasSGEMM>(ctx,trans_a,trans_b,M,N,K,bias,act,im2col_chan);
		return g;
#else
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new StandardSGEMM(ctx,trans_a,trans_b,M,N,K,bias,act,im2col_chan));
        return g;
#endif
    }
    std::unique_ptr<GEMM> GEMM::get_optimal_conv_gemm(
            Context &ctx,DataType dtype,
            GemmOpMode op_mode,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int kernel[2],int dilate[2],int padding[2],int stride[2],int groups,
            int src_channels,int src_rows,int src_cols,
            int tgt_rows,int tgt_cols,
            int bias,
            StandardActivations act,
            int im2col_chan)
    {
#if 1
		DLPRIM_CHECK(dtype == float_data); // for now, this will be made different later!
		std::unique_ptr<GEMM> g = std::make_unique<BlasConvSGEMM>(ctx, op_mode,
			trans_a, trans_b, M, N, K, kernel,dilate, padding,stride, groups,
			src_channels, src_rows, src_cols, tgt_rows, tgt_cols, bias,act,im2col_chan);
		return g;
#else
        DLPRIM_CHECK(dtype == float_data);
        std::unique_ptr<GEMM> g(new ConvSGEMM(ctx,op_mode,
            trans_a,trans_b,M,N,K,
            kernel,dilate,padding,stride,groups,
            src_channels,src_rows,src_cols,
            tgt_rows,tgt_cols,
            bias,act,im2col_chan));
        return g;
#endif
    }

    void GEMM::batch_sgemm(DataType dt,
                          bool trans_a,bool trans_b,
                          int Batch, // number of matrices
                          int M,int N,int K,
#if VULKAN_API
                          tart::buffer_ptr &a,
                          uint32_t offset_a, 
                          int batch_stride_a,
                          int lda,
                          tart::buffer_ptr &b,
                          uint32_t offset_b,
                          int batch_stride_b,
                          int ldb,
                          tart::buffer_ptr &c,
                          uint32_t offset_c,
#else
                          cl::Buffer &a,
                          cl_ulong offset_a, 
                          int batch_stride_a,
                          int lda,
                          cl::Buffer &b,
                          cl_ulong offset_b,
                          int batch_stride_b,
                          int ldb,
                          cl::Buffer &c,
                          cl_ulong offset_c,
#endif
                          int batch_stride_c,
                          int ldc,
                          float beta,
                          ExecutionContext const &e)
    {
#if VULKAN_API
		DLPRIM_CHECK(dt == float_data);
        
        StandardActivations act = StandardActivations::identity;
        Context ctx(e);
        BlasBatchSGEMM gemm_opt(ctx,trans_a,trans_b,M,N,K,act);
        gemm_opt.gemm(Batch,M,N,K,
                a,offset_a,batch_stride_a,lda,
                b,offset_b,batch_stride_b,ldb,
                c,offset_c,batch_stride_c,ldc,
                beta,
                e);
#else
        DLPRIM_CHECK(dt == float_data);
        
        StandardActivations act = StandardActivations::identity;
        Context ctx(e);
        BatchSGEMM gemm_opt(ctx,trans_a,trans_b,M,N,K,act);
        gemm_opt.gemm(Batch,M,N,K,
                a,offset_a,batch_stride_a,lda,
                b,offset_b,batch_stride_b,ldb,
                c,offset_c,batch_stride_c,ldc,
                beta,
                e);
#endif
    }



} // gpu
} // dlprim 
