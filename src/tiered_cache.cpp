#include <dlprim/gpu/tiered_cache.hpp>
#include <dlprim/gpu/program_cache.hpp>

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
	}
	else if (dtypes.size() == 1)
	{
		tart::DType dt = dtypes[0];
		
		mAxpbyProgram = gpu::Cache::instance().get_program(device, "axpby");
		
		mCopyProgram = gpu::Cache::instance().get_program(device, "copy", "dtype", dt.glsl());
		
		mBnSumsProgram = gpu::Cache::instance().get_program(device, "bn_sums", "dtype", dt.glsl());
		mBnUtilsProgram = gpu::Cache::instance().get_program(device, "bn_utils", "dtype", dt.glsl());
		
		mBwdBiasProgram = gpu::Cache::instance().get_program(device, "bwd_bias", "dtype", dt.glsl());
				
		mCol2imProgram = Cache::instance().get_program(device, "col2im_torch", "dtype", dt.glsl());
		
		mFwdBiasProgram = gpu::Cache::instance().get_program(device, "fwd_bias", "dtype", dt.glsl());
		
		mGlobalPoolingProgram = gpu::Cache::instance().get_program(device, "global_pooling", "dtype", dt.glsl());
		
		mIm2colProgram = Cache::instance().get_program(device, "im2col_torch", "dtype", dt.glsl());
		
		mPoolingProgram = gpu::Cache::instance().get_program(device, "pooling", "dtype", dt.glsl());
		
		mRandomProgram = Cache::instance().get_program(device, "random", "dtype", dt.glsl());
		mScalProgram = Cache::instance().get_program(device, "random", "dtype", dt.glsl());
		mSpatialSoftmaxProgram = Cache::instance().get_program(device, "spatial_softmax_torch", "dtype", dt.glsl());
	}
	else if (dtypes.size() == 2)
	{
		#if 1
			bool use_io_type = (dtypes[0] == dtypes[1]);
			tart::DType dt0 = dtypes[0];
			tart::DType dt1 = dtypes[1];
		#else
			bool use_io_type = (dtypes[0] == dtypes[1]);
			tart::DType dt0 = data_type_to_tart_dtype(dtypes[0], use_io_type);
			tart::DType dt1 = data_type_to_tart_dtype(dtypes[1], use_io_type);
		#endif
		
		// avoid compilation errors
		if (dt1.isFloatingPoint())
		{
			mCopyStridedProgram = gpu::Cache::instance().get_program(device, "copy_strided", "dtype_src", dt0.glsl(), "dtype_tgt", dt1.glsl() );
		}
		mNullLossBwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_bwd", "dtype", dt0.glsl(), "itype", dt1.glsl());
		mNullLossFwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_fwd", "dtype", dt0.glsl(), "itype", dt1.glsl());
	}
}

} // namespace gpu

} // namespace dlprim
