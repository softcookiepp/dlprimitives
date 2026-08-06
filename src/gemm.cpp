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

namespace dlprim
{
namespace gpu
{

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
		tart::device_ptr mDevice = nullptr;
		
		// TODO: make these all stateless.
		clblast::Transpose mATrans;
		clblast::Transpose mBTrans;
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
		throw std::runtime_error("This isn't ported to Vulkan yet, and I doubt it ever will be due to lack of design simplicity");
		return nullptr;
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
