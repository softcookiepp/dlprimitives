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

// The fundamental distinguishing factor of each pointwise op.
typedef std::tuple<
	std::vector<tart::DType>, // xtypes
	std::vector<tart::DType>, // ytypes
	size_t, // weight count
	std::string // code
> PointwiseOpKey;

class PointwiseCache
{
	tart::device_ref mDevice;
	std::map<PointwiseOpKey, tart::program_ptr> mPointwisePrograms;
	//std::vector<std::pair<PointwiseOpKey, tart::program_ptr>> mPointwisePrograms;
	
	tart::program_ptr findProgram(const PointwiseOpKey& key);
public:
	PointwiseCache(const tart::device_ptr& device);
	
	tart::program_ptr getPointwiseOperation(std::vector<Tensor>& xs,
		std::vector<Tensor>& ys, std::vector<double> ws, const std::string& code);
		
	tart::program_ptr getPointwiseBroadcastOperation(
		const std::vector<Tensor>& xs,
		const std::vector<Tensor>& ys,
		const std::vector<double>& ws,
		const std::vector<tart::DType>& dts,
		const std::string &code,
		const bool shrinkDims);
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
	
	tart::program_ptr mSpatialSoftmaxProgram = nullptr;
	
public:
	AllPrograms(const tart::device_ptr& device, const std::vector<tart::DType>& dtypes);
	
	friend class PerDeviceProgramCache;
};

class ProgramsPerDtypes
{
	tart::device_ref mDevice;
	// use enum for faster lookup
	std::map<std::vector<tart::DataType>, std::unique_ptr<AllPrograms>> mAllPrograms;
public:
	ProgramsPerDtypes(const tart::device_ptr& device);
	AllPrograms& getAllPrograms(const std::vector<tart::DType>& dtypes);
};

class PerDeviceProgramCache
{
	std::map<std::uintptr_t, std::unique_ptr<ProgramsPerDtypes>> mProgramsPerDtypes;
	std::map<std::uintptr_t, std::unique_ptr<PointwiseCache>> mPointwiseCaches;
	
	PointwiseCache& getPointwiseCache(const tart::device_ptr& device);
public:
	AllPrograms& getAllPrograms(const tart::device_ptr& device, const std::vector<tart::DType>& dtypes);
	
	static PerDeviceProgramCache& instance();
	
	tart::program_ptr getPointwiseOperation(const tart::device_ptr& device, std::vector<Tensor>& xs,
		std::vector<Tensor>& ys, std::vector<double> ws, const std::string& code);
	
	tart::program_ptr getPointwiseBroadcastOperation(
		const tart::device_ptr& device,
		const std::vector<Tensor>& xs,
		const std::vector<Tensor>& ys,
		const std::vector<double>& ws,
		const std::vector<tart::DType>& dts,
		const std::string &code,
		const bool shrinkDims);
	
	inline const tart::program_ptr& activation(const tart::device_ptr& device) { return getAllPrograms(device, {}).mActivationProgram; }
	inline const tart::program_ptr& axpby(const tart::device_ptr& device) { return getAllPrograms(device, {}).mAxpbyProgram; }
	
	inline const tart::program_ptr& bn_sums(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mBnSumsProgram; }
	inline const tart::program_ptr& bn_utils(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mBnUtilsProgram; }
	
	inline const tart::program_ptr& bwd_bias(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mBwdBiasProgram; }
	
	inline const tart::program_ptr& copy(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mCopyProgram; }
	
	inline const tart::program_ptr& copy_strided(const tart::device_ptr& device, const tart::DType dtSrc, const tart::DType dtTgt) { return getAllPrograms(device, {dtSrc, dtTgt}).mCopyStridedProgram; }
	
	inline const tart::program_ptr& col2im(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mCol2imProgram; }
	
	inline const tart::program_ptr& fwd_bias(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mFwdBiasProgram; }
	
	inline const tart::program_ptr& im2col(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mIm2colProgram; }
	
	inline const tart::program_ptr& interpolate2d(const tart::device_ptr& device) { return getAllPrograms(device, {}).mInterpolate2dProgram; }
	
	inline const tart::program_ptr& nll_loss_bwd(const tart::device_ptr& device, const tart::DType dt, const tart::DType it) { return getAllPrograms(device, {dt, it}).mNullLossBwdProgram; }
	inline const tart::program_ptr& nll_loss_fwd(const tart::device_ptr& device, const tart::DType dt, const tart::DType it) { return getAllPrograms(device, {dt, it}).mNullLossFwdProgram; }
	
	inline const tart::program_ptr& pooling(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mPoolingProgram; }
	inline const tart::program_ptr& global_pooling(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mGlobalPoolingProgram; }
	
	inline const tart::program_ptr& random(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mRandomProgram; }
	inline const tart::program_ptr& scal(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mScalProgram; }
	
	inline const tart::program_ptr& spatial_softmax(const tart::device_ptr& device, const tart::DType dtype) { return getAllPrograms(device, {dtype}).mSpatialSoftmaxProgram; }
};

} // namespace gpu

} // namespace dlprim
