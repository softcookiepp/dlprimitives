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

AllPrograms& PerDeviceProgramCache::getAllPrograms(const tart::device_ptr& device, const std::vector<DataType>& dtypes)
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

AllPrograms& ProgramsPerDtypes::getAllPrograms(const std::vector<DataType>& dtypes)
{
	if (mAllPrograms.find(dtypes) == mAllPrograms.end())
	{
		mAllPrograms[dtypes] = std::make_unique<AllPrograms>(mDevice.lock(), dtypes);
	}
	return *mAllPrograms[dtypes];
}

AllPrograms::AllPrograms(const tart::device_ptr& device, const std::vector<DataType>& dtypes) :
	mDevice(device)
{
	// Once this is created, initialize as many as possible from just the data
	Context ctx(device);
	
	if (dtypes.size() == 0)
	{
		// any programs that don't require a dtype.
		// Some of them just haven't been made compatible with anything other than floats, in which case they will be moved to the next section later
		mActivationProgram = gpu::Cache::instance().get_program(ctx,"activation");
		mAxpbyProgram = gpu::Cache::instance().get_program(ctx,"axpby");
		
		mInterpolate2dProgram = gpu::Cache::instance().get_program(ctx, "interpolate_2d");
		
		mPoolingProgram = gpu::Cache::instance().get_program(ctx, "pooling");
		mGlobalPoolingProgram = gpu::Cache::instance().get_program(ctx, "global_pooling");
	}
	else if (dtypes.size() == 1)
	{
		tart::DType dt = data_type_to_tart_dtype(dtypes[0]);
		mCopyProgram = gpu::Cache::instance().get_program(ctx, "copy", "dtype", dt.glsl());
		
		mBnSumsProgram = gpu::Cache::instance().get_program(ctx, "bn_sums", "dtype", dt.glsl());
		mBnUtilsProgram = gpu::Cache::instance().get_program(ctx, "bn_utils", "dtype", dt.glsl());
		
		mBwdBiasProgram = gpu::Cache::instance().get_program(ctx, "bwd_bias", "dtype", dt.glsl());
				
		mCol2imProgram = Cache::instance().get_program(ctx, "col2im_torch", "dtype", dt.glsl());
		
		mFwdBiasProgram = gpu::Cache::instance().get_program(ctx, "fwd_bias", "dtype", dt.glsl());
		
		mIm2colProgram = Cache::instance().get_program(ctx, "im2col_torch", "dtype", dt.glsl());
		
		mRandomProgram = Cache::instance().get_program(ctx, "random", "dtype", dt.glsl());
		mScalProgram = Cache::instance().get_program(ctx, "random", "dtype", dt.glsl());
	}
	else if (dtypes.size() == 2)
	{
		bool use_io_type = (dtypes[0] == dtypes[1]);
		tart::DType dt0 = data_type_to_tart_dtype(dtypes[0], use_io_type);
		tart::DType dt1 = data_type_to_tart_dtype(dtypes[1], use_io_type);
		mCopyStridedProgram = gpu::Cache::instance().get_program(ctx,"copy_strided", "dtype_src", dt0.glsl(), "dtype_tgt", dt1.glsl() );
	}
}

} // namespace gpu

} // namespace dlprim
