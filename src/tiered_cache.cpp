#include <dlprim/gpu/tiered_cache.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim
{

namespace gpu
{

AllPrograms& PerDeviceProgramCache::getAllPrograms(const tart::device_ptr& device)
{
	std::uintptr_t key = (std::uintptr_t)device.get();
	if (mAllProgramsPerDevice.find(key) == mAllProgramsPerDevice.end()
	{
		// create new AllPrograms
		mAllProgramsPerDevice[key] = std::make_unique<AllPrograms>(device);
	}
	return mAllProgramsPerDevice[key];
}

AllPrograms::AllPrograms(const tart::device_ptr& device) :
	mDevice(device)
{
	// Once this is created, initialize as many as possible from just the data
}

} // namespace gpu

} // namespace dlprim
