#pragma once
#include <dlprim/context.hpp>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace dlprim
{

namespace gpu
{
	
class PointwiseCache
{
public:
	PointwiseCache(const tart::device_ptr& device);
	
};

class AllPrograms
{
	tart::device_ref mDevice;
public:
	AllPrograms(const tart::device_ptr& device);
};

class PerDeviceProgramCache
{
	std::map<std::uintptr_t, std::unique_ptr<AllPrograms>> mAllProgramsPerDevice;
public:
	AllPrograms& getAllPrograms(const tart::device_ptr& device);
};

} // namespace gpu

} // namespace dlprim
