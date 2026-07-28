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
		// any programs that don't require a dtype
		mPoolingProgram = gpu::Cache::instance().get_program(ctx, "pooling");
		
		// Ok, so the changes I made to global pooling actually break something.
		// This will require more investigation...
		mGlobalPoolingProgram = gpu::Cache::instance().get_program(ctx, "global_pooling");
	}
	else if (dtypes.size() == 1)
	{
		
	}
}

} // namespace gpu

} // namespace dlprim
