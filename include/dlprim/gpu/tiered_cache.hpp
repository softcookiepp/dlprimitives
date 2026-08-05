#pragma once
#include <dlprim/context.hpp>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <vector>

namespace dlprim
{

namespace gpu
{
	
class PointwiseCache
{
public:
	PointwiseCache(const tart::device_ptr& device);
	
};

class PerDeviceProgramCache;

class AllPrograms
{
	tart::device_ref mDevice;
	
	tart::program_ptr mActivationProgram = nullptr;
	tart::program_ptr mAxpbyProgram = nullptr;
	
	tart::program_ptr mBnSumsProgram = nullptr;
	tart::program_ptr mBnUtilsProgram = nullptr;
	
	tart::program_ptr mBwdBiasProgram = nullptr;
	
	tart::program_ptr mCopyProgram = nullptr;
	
	tart::program_ptr mCopyStridedProgram = nullptr;
	
	tart::program_ptr mCol2imProgram = nullptr;
	
	tart::program_ptr mFwdBiasProgram = nullptr;
	
	tart::program_ptr mIm2colProgram = nullptr;
	
	tart::program_ptr mInterpolate2dProgram = nullptr;
	
	tart::program_ptr mNullLossBwdProgram = nullptr;
	tart::program_ptr mNullLossFwdProgram = nullptr;
	
	tart::program_ptr mPoolingProgram = nullptr;
	tart::program_ptr mGlobalPoolingProgram = nullptr;
	
	tart::program_ptr mRandomProgram = nullptr;
	
	tart::program_ptr mScalProgram = nullptr;
	
public:
	AllPrograms(const tart::device_ptr& device, const std::vector<DataType>& dtypes);
	
	friend class PerDeviceProgramCache;
};

class ProgramsPerDtypes
{
	tart::device_ref mDevice;
	std::map<std::vector<DataType>, std::unique_ptr<AllPrograms>> mAllPrograms;
public:
	ProgramsPerDtypes(const tart::device_ptr& device);
	AllPrograms& getAllPrograms(const std::vector<DataType>& dtypes);
};

class PerDeviceProgramCache
{
	std::map<std::uintptr_t, std::unique_ptr<ProgramsPerDtypes>> mProgramsPerDtypes;
public:
	AllPrograms& getAllPrograms(const tart::device_ptr& device, const std::vector<DataType>& dtypes);
	
	static PerDeviceProgramCache& instance();
	
	inline const tart::program_ptr& activation(const tart::device_ptr& device) { return getAllPrograms(device, {}).mActivationProgram; }
	inline const tart::program_ptr& axpby(const tart::device_ptr& device) { return getAllPrograms(device, {}).mAxpbyProgram; }
	
	inline const tart::program_ptr& bn_sums(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mBnSumsProgram; }
	inline const tart::program_ptr& bn_utils(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mBnUtilsProgram; }
	
	inline const tart::program_ptr& bwd_bias(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mBwdBiasProgram; }
	
	inline const tart::program_ptr& copy(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mCopyProgram; }
	
	inline const tart::program_ptr& copy_strided(const tart::device_ptr& device, const DataType dtSrc, const DataType dtTgt) { return getAllPrograms(device, {dtSrc, dtTgt}).mCopyStridedProgram; }
	
	inline const tart::program_ptr& col2im(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mCol2imProgram; }
	
	inline const tart::program_ptr& fwd_bias(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mFwdBiasProgram; }
	
	inline const tart::program_ptr& im2col(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mIm2colProgram; }
	
	inline const tart::program_ptr& interpolate2d(const tart::device_ptr& device) { return getAllPrograms(device, {}).mInterpolate2dProgram; }
	
	inline const tart::program_ptr& nll_loss_bwd(const tart::device_ptr& device, const DataType dt, const DataType it) { return getAllPrograms(device, {dt, it}).mNullLossBwdProgram; }
	
	inline const tart::program_ptr& pooling(const tart::device_ptr& device) { return getAllPrograms(device, {}).mPoolingProgram; }
	inline const tart::program_ptr& global_pooling(const tart::device_ptr& device) { return getAllPrograms(device, {}).mGlobalPoolingProgram; }
	
	inline const tart::program_ptr& random(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mRandomProgram; }
	inline const tart::program_ptr& scal(const tart::device_ptr& device, const DataType dtype) { return getAllPrograms(device, {dtype}).mScalProgram; }
};

} // namespace gpu

} // namespace dlprim
