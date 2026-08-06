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
#include <clblast_vk.h>
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
			if(M >= 256 && N >= 256) {
				tile_size_m_ = 128;
				tile_size_n_ = 128;
				block_size_m_ = 8;
				block_size_n_ = 8;
				tile_size_k_ = 16;
				off_ = 1;
			}
			else if(M >= 128 && N>= 128) {
				tile_size_m_ = 64;
				tile_size_n_ = 64;
				block_size_m_ = 8;
				block_size_n_ = 8;
				tile_size_k_ = 16;
				off_ = 1;
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
#if 0
            if(!batch_gemm_) {
                int cores = ctx.estimated_core_count();
                if(cores >= 256 && M * N / (block_size_m_ * block_size_n_) < 4 * cores && K > M*16 && K > N*16) {
                    reduce_k_ = 8;
                    set_scale(ctx,activation);
                }
            }
#endif
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
			#if 1
				throw std::runtime_error("this shouldn't be used");
			#else
				if(sep_scale_ == false) {
					tart::program_ptr
						prog = gpu::Cache::instance().get_program(ctx,"scal");
					scal_ = prog->getKernel("sscal");
					if(activation != StandardActivations::identity) {
						tart::program_ptr
							prog = gpu::Cache::instance().get_program(ctx,"activation",
															"ACTIVATION",int(activation));
						tart::kernel_ptr k = prog->getKernel("activation");
						act_ = k;
						activation = StandardActivations::identity;
						sep_act_ = true;
					}

					sep_scale_ = true;
				}
            #endif
        }

        void activation(size_t size, 
			tart::buffer_ptr x,
			size_t x_offset,
			ExecutionContext const &ec)
        {
			act_->setArg(0, uint32_t(size));
			act_->setArg(1, x);
			act_->setArg(2, x_offset);
			act_->setArg(3, x);
			act_->setArg(4, x_offset);
			// TODO: ensure the global size for this isn't wrong
			act_->enqueue({size}, {1});
        }
        void scale(size_t size,float s, const tart::buffer_ptr& x, uint32_t x_offset,ExecutionContext const &ec)
        {
            int wg = 64;
            if(size >= 1024)
                wg = 256;
            int p=0;
            scal_->setArg(p++, uint32_t(size));
            scal_->setArg(p++, s);
            scal_->setArg(p++, x);
            scal_->setArg(p++, x_offset);
			std::vector<uint32_t> l({wg});
			std::vector<uint32_t> g = gpu::round_range(size, l);
			g[0] = g[0]/wg;
			g.resize(3, 1);
			scal_->enqueue(g, {});
        }
        int tile_size_n_,tile_size_m_,tile_size_k_;
        int block_size_n_,block_size_m_;
        int off_;
        int reduce_k_;
        bool sep_scale_;
        bool sep_act_;
        bool batch_gemm_;
        tart::kernel_ptr scal_;
        tart::kernel_ptr act_;
        bool zorder_ = false;
        
        StandardActivations mActivation;
    };

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
			const float alpha = 1.0;
			clblast::GemmStridedBatched(clblast::Layout::kRowMajor, mATrans, mBTrans,
				M, N, K, alpha,
				a, 0, lda, batch_stride_a,
				b, 0, ldb, batch_stride_b,
				beta,
				c, 0, ldc, batch_stride_c,
				batches, mDevice);
		}
	};

	// dlprim's existing GEMM kernels are too difficult for me to port to GLSL.
	// So I am simply taking the stuff I developed for CLBlast and putting it here!
	
	class BlasSGEMM : public GEMM
	{
		
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
			const float alpha = 1.0;
			if (beta == 0.0 && mUseBias)
			{
				// adjust beta, as the bias will not work unless beta is 1.0
				beta = 1.0;
			}
			
			if (mUseBias)
			{
				// Copy bias to C in preparation for biasing
				for (size_t i = 0; i < M; i += 1)
				{
					uint32_t c_row_offset = (ldc*i) + offset_c;
					// pretty sure x_inc is just 1, unless C somehow has strides.
					clblast::Copy<float>(N, bias, bias_offset, 1,
						c, c_row_offset, 1, mDevice);
				}
			}
			
			clblast::Gemm(clblast::Layout::kRowMajor, mATrans, mBTrans, M, N, K, alpha,
				a, offset_a, lda, b, offset_b, ldb, beta, c, offset_c, ldc, mDevice);
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

    std::unique_ptr<GEMM> GEMM::get_optimal_gemm(
            Context &ctx,DataType dtype,
            bool trans_a,bool trans_b,
            int M,int N,int K,
            int bias,
            StandardActivations act,
            int im2col_chan)
    {
		DLPRIM_CHECK(dtype == float_data);
		std::unique_ptr<GEMM> g = std::make_unique<BlasSGEMM>(ctx,trans_a,trans_b,M,N,K,bias,act,im2col_chan);
		return g;
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
		// TODO: remove or something. I don't know how this will work :c
		#if 1
			throw std::runtime_error("YOU SHALL NOT PASS");
			return nullptr;
		#else
			DLPRIM_CHECK(dtype == float_data); // for now, this will be made different later!
			std::unique_ptr<GEMM> g = std::make_unique<BlasConvSGEMM>(ctx, op_mode,
				trans_a, trans_b, M, N, K, kernel,dilate, padding,stride, groups,
				src_channels, src_rows, src_cols, tgt_rows, tgt_cols, bias,act,im2col_chan);
			return g;
		#endif
    }

    void GEMM::batch_sgemm(DataType dt,
                          bool trans_a,bool trans_b,
                          int Batch, // number of matrices
                          int M,int N,int K,
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
                          int batch_stride_c,
                          int ldc,
                          float beta,
                          ExecutionContext const &e)
    {
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
    }



} // gpu
} // dlprim 
