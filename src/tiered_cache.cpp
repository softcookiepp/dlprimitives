#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <sstream>
#include <iostream>

namespace dlprim
{

namespace gpu
{
	
PerDeviceProgramCache& PerDeviceProgramCache::instance()
{
	static PerDeviceProgramCache cache;
	return cache;
}

AllPrograms& PerDeviceProgramCache::getAllPrograms(const tart::device_ptr& device, const std::vector<tart::DType>& dtypes)
{
	std::uintptr_t key = (std::uintptr_t)device.get();
	if (mProgramsPerDtypes.find(key) == mProgramsPerDtypes.end() )
	{
		// create new AllPrograms
		mProgramsPerDtypes[key] = std::make_unique<ProgramsPerDtypes>(device);
	}
	return mProgramsPerDtypes[key]->getAllPrograms(dtypes);
}

ProgramsPerDtypes::ProgramsPerDtypes(const tart::device_ptr& device) :
	mDevice(device)
{
}

AllPrograms& ProgramsPerDtypes::getAllPrograms(const std::vector<tart::DType>& dtypes)
{
	std::vector<tart::DataType> dtEnums(dtypes.size());
	for (size_t i = 0; i < dtypes.size(); i += 1)
	{
		dtEnums[i] = dtypes[i]();
	}
	if (mAllPrograms.find(dtEnums) == mAllPrograms.end())
	{
		mAllPrograms[dtEnums] = std::make_unique<AllPrograms>(mDevice.lock(), dtypes);
	}
	return *mAllPrograms[dtEnums];
}

AllPrograms::AllPrograms(const tart::device_ptr& device, const std::vector<tart::DType>& dtypes) :
	mDevice(device)
{
	if (dtypes.size() == 0)
	{
		// any programs that don't require a dtype.
		// Some of them just haven't been made compatible with anything other than floats, in which case they will be moved to the next section later
		mActivationProgram = gpu::Cache::instance().get_program(device, "activation");
		
		mInterpolate2dProgram = gpu::Cache::instance().get_program(device, "interpolate_2d");
		
		mBiasProgram = gpu::Cache::instance().get_program(device, "bias");
	}
	else if (dtypes.size() == 1)
	{
		tart::DType dt = dtypes[0];
		
		mAxpbyProgram = gpu::Cache::instance().get_program(device, "axpby");
		
		mCopyProgram = gpu::Cache::instance().get_program(device, "copy", "DTYPE", dt.get());
		
		mBnSumsProgram = gpu::Cache::instance().get_program(device, "bn_sums", "DTYPE", dt.get());
		mBnUtilsProgram = gpu::Cache::instance().get_program(device, "bn_utils", "DTYPE", dt.get());
		
		mBwdBiasProgram = gpu::Cache::instance().get_program(device, "bwd_bias", "DTYPE", dt.get());
				
		mCol2imProgram = Cache::instance().get_program(device, "col2im_torch", "DTYPE", dt.get());
		
		mFwdBiasProgram = gpu::Cache::instance().get_program(device, "fwd_bias", "DTYPE", dt.get());
		
		mGlobalPoolingProgram = gpu::Cache::instance().get_program(device, "global_pooling", "DTYPE", dt.get());
		
		mIm2colProgram = Cache::instance().get_program(device, "im2col_torch", "DTYPE", dt.get());
		
		mPoolingProgram = gpu::Cache::instance().get_program(device, "pooling", "DTYPE", dt.get());
		
		mRandomProgram = Cache::instance().get_program(device, "random", "DTYPE", dt.get());
		mScalProgram = Cache::instance().get_program(device, "random", "DTYPE", dt.get());
	}
	else if (dtypes.size() == 2)
	{
		bool use_io_type = (dtypes[0] == dtypes[1]);
		tart::DType dt0 = dtypes[0];
		tart::DType dt1 = dtypes[1];
		
		// avoid compilation errors
		if (dt1.isFloatingPoint())
		{
			mCopyStridedProgram = gpu::Cache::instance().get_program(device, "copy_strided", "DTYPE_SRC", dt0.get(), "DTYPE_TGT", dt1.get() );
		}
		if (dt0.isFloatingPoint() && !dt1.isFloatingPoint())
		{
			std::cout << "dt1: " << dt1.glsl() << std::endl;
			mNullLossBwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_bwd", "DTYPE", dt0.get(), "ITYPE", dt1.get());
			mNullLossFwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_fwd", "DTYPE", dt0.get(), "ITYPE", dt1.get());
		}
		
		mPointwiseUnaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-unary-unary", "TYPEOF_X0", dt0.get(), "TYPEOF_Y0", dt1.get());
		
		mPointwiseReduceUnaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-reduce-unary-unary",
			"TYPEOF_X0", dt0.get(), "TYPEOF_Y0", dt1.get());
			
		if (dt0.isFloatingPoint() && dt1.isFloatingPoint())
		{
			mSoftmaxProgram = gpu::Cache::instance().get_program(device, "softmax2",
				"TYPEOF_X0", dt0.get(), "TYPEOF_Y0", dt1.get());
			
			mSoftmaxBwdProgram = gpu::Cache::instance().get_program(device, "softmax2-bwd",
				"TYPEOF_X0", dt0.get(), "TYPEOF_Y0", dt1.get());
		}
	}
	else if(dtypes.size() == 3)
	{
		// Disable, as it currently does not compile
		// mGemm2Program = gpu::Cache::instance().get_program(device, "gemm2", "A_TYPE", dtypes[0].get(), "B_TYPE", dtypes[1].get(), "D_TYPE", dtypes[2].get());
		
		mPointwiseReduceBinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-reduce-binary-unary",
			"TYPEOF_X0", dtypes[0].get(), "TYPEOF_X1", dtypes[1].get(), "TYPEOF_Y0", dtypes[2].get());
		
		mPointwiseReduceUnaryBinaryProgram = gpu::Cache::instance().get_program(device, "pointwise-reduce-unary-binary",
			"TYPEOF_X0", dtypes[0].get(), "TYPEOF_Y0", dtypes[1].get(), "TYPEOF_Y1", dtypes[2].get());
		
		mPointwiseBinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-binary-unary",
			"TYPEOF_X0", dtypes[0].get(),
			"TYPEOF_X1", dtypes[1].get(),
			"TYPEOF_Y0", dtypes[2].get());
		
		mPointwiseUnaryBinaryProgram = gpu::Cache::instance().get_program(device, "pointwise-unary-binary",
			"TYPEOF_X0", dtypes[0].get(),
			"TYPEOF_Y0", dtypes[1].get(),
			"TYPEOF_Y1", dtypes[2].get());
	}
	else if (dtypes.size() == 4)
	{
		mPointwiseTrinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-trinary-unary",
			"TYPEOF_X0", dtypes[0].get(),
			"TYPEOF_X1", dtypes[1].get(),
			"TYPEOF_X2", dtypes[2].get(),
			"TYPEOF_Y0", dtypes[3].get());
		
		mPointwiseReduceTrinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-reduce-trinary-unary",
			"TYPEOF_X0", dtypes[0].get(),
			"TYPEOF_X1", dtypes[1].get(),
			"TYPEOF_X2", dtypes[2].get(),
			"TYPEOF_Y0", dtypes[3].get());
	}
	else if (dtypes.size() == 5)
	{
		mPointwiseReduceQuaternaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-reduce-quaternary-unary",
			"TYPEOF_X0", dtypes[0].get(),
			"TYPEOF_X1", dtypes[1].get(),
			"TYPEOF_X2", dtypes[2].get(),
			"TYPEOF_X3", dtypes[3].get(),
			"TYPEOF_Y0", dtypes[4].get());
	}
}

} // namespace gpu

} // namespace dlprim
