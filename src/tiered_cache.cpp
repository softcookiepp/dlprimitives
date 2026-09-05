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

PointwiseCache::PointwiseCache(const tart::device_ptr& device) : mDevice(device) {}

tart::program_ptr PointwiseCache::findProgram(const PointwiseOpKey& k)
{
	for (auto& pair : mPointwisePrograms)
	{
		if (pair.first == k)
		{
			return pair.second;
		}
	}
	return nullptr;
}

PointwiseOpKey makeKey(
	const std::vector<Tensor>& xs,
	const std::vector<Tensor>& ys,
	const std::vector<double>& ws,
	const std::vector<tart::DType>& dts,
	const std::string &code,
	const bool shrinkDims)
{
	PointwiseOpKey k;
	std::get<0>(k).resize(xs.size());
	std::get<1>(k).resize(ys.size());
	std::get<2>(k) = ws.size();
	for (size_t i = 0; i < xs.size(); i += 1)
	{
		std::get<0>(k)[i] = xs[i].dtype();
	}
	for (size_t i = 0; i < std::get<1>(k).size(); i += 1)
	{
		std::get<1>(k)[i] = ys[i].dtype();
	}
	std::get<3>(k) = dts;
	std::get<4>(k) = code;
	std::get<5>(k) = shrinkDims;
	std::get<6>(k) = "";
	std::get<7>(k) = "";
	return k;
}

PointwiseOpKey makeKey(
	const std::vector<TensorSpecs>& xs,
	const std::vector<TensorSpecs>& ys,
	size_t weightCount,
	const std::vector<tart::DType>& dts,
	const std::string &code,
	const bool shrinkDims,
	const std::string& reduceInitCode = "",
	const std::string& reduceCode = "")
{
	PointwiseOpKey k;
	std::get<0>(k).resize(xs.size());
	std::get<1>(k).resize(ys.size());
	std::get<2>(k) = weightCount;
	for (size_t i = 0; i < xs.size(); i += 1)
	{
		std::get<0>(k)[i] = xs[i].dtype();
	}
	for (size_t i = 0; i < std::get<1>(k).size(); i += 1)
	{
		std::get<1>(k)[i] = ys[i].dtype();
	}
	std::get<3>(k) = dts;
	std::get<4>(k) = code;
	std::get<5>(k) = shrinkDims;
	std::get<6>(k) = reduceInitCode;
	std::get<7>(k) = reduceCode;
	return k;
}

tart::program_ptr PointwiseCache::getPointwiseBroadcastReduceOperation(
	std::vector<TensorSpecs>& xs,
	std::vector<TensorSpecs>& ys,
	int weights_count,
	const tart::DType& weights_type,
	const std::string& compute_code,
	const std::string& reduce_init,
	const std::string& reduce)
{
	DLPRIM_CHECK(!xs.empty());
	DLPRIM_CHECK(!ys.empty());
	
	std::vector<tart::DType> dts(1, weights_type);
	PointwiseOpKey k = makeKey(xs, ys, weights_count, dts, compute_code, false, reduce_init, reduce);
	
	if (mPointwisePrograms.find(k) == mPointwisePrograms.end())
	{
		// all the defines
		std::ostringstream PARAMS,PREPARE_LOAD_INPUT_ALL,REDUCE_INIT_ALL,LOAD_INPUT_ALL,
			LOAD_REDUCE_ALL,SAVE_REDUCE_ALL,LOAD_REDUCED_SAVE_GLOBAL_ALL;
		std::stringstream BUFFER_DEFS;
		std::stringstream BUFFER_OFFSETS; // in order to emulate pointer math
		std::stringstream TYPE_DEFS;
		std::stringstream REDUCE_INIT_SHARED;
		size_t bindIndex = 0;
		for(size_t i=0;i<xs.size();i++) {
			std::string type = xs[i].dtype().glsl();
			std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
			std::string suffix_buf = "(" + type + "," + std::to_string(i) + ", " + std::to_string(bindIndex) + ") ";
			TYPE_DEFS << "#define typeof_x" << i << " " << type << "\n";
			BUFFER_DEFS << "PARAM_INPUT_BUF" << suffix_buf;
			BUFFER_OFFSETS << "PARAM_INPUT_BUF_OFFSET" << suffix;
			PARAMS << "PARAM_INPUT" << suffix;
			PREPARE_LOAD_INPUT_ALL << "PREPARE_LOAD_INPUT" << suffix << ";\\\n";
			LOAD_INPUT_ALL << "LOAD_INPUT(" << i << ");\\\n";
			bindIndex += 1;
		}

		for(size_t i=0;i<ys.size();i++) {
			std::string type = ys[i].dtype().glsl();
			std::string ptype = ys[i].dtype().glsl();
			std::string suffix_out = "(" + type + "," + ptype + "," + std::to_string(i) + ") ";
			std::string suffix_out_buf = "(" + type + "," + ptype + "," + std::to_string(i) + ", " + std::to_string(bindIndex) + ") ";
			std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
			TYPE_DEFS << "#define typeof_y" << i << " " << type << "\n";
			BUFFER_DEFS << "PARAM_OUTPUT_BUF" << suffix_out_buf;
			BUFFER_OFFSETS << "PARAM_OUTPUT_BUF_OFFSET" << suffix_out;
			PARAMS << "PARAM_OUTPUT" << suffix_out;
			REDUCE_INIT_SHARED << "REDUCE_INIT"<<suffix << ";";
			LOAD_REDUCE_ALL << "LOAD_REDUCE("<<i<<");\\\n";
			SAVE_REDUCE_ALL << "SAVE_REDUCE("<<i<<");\\\n";
			LOAD_REDUCED_SAVE_GLOBAL_ALL << "LOAD_REDUCED_SAVE_GLOBAL("<<i<<");\\\n";
			bindIndex += 1;
		}
		
		REDUCE_INIT_ALL << core::format_code(reduce_init) << "\n";
		for(size_t i=0;i< weights_count;i++) {
			std::string type = weights_type.glsl();
			PARAMS << type << " w" << i <<"; ";
			TYPE_DEFS << "#define typeof_w" << i << " " << type << "\n";
		}

		mPointwisePrograms[k] = gpu::Cache::instance().get_program(
			mDevice.lock(), "pointwise_broadcast_reduce",
			"$TYPE_DEFS", TYPE_DEFS.str(),
			"#BUFFER_DEFS", BUFFER_DEFS.str(),
			"#BUFFER_OFFSETS", BUFFER_OFFSETS.str(),
			"#PARAMS",PARAMS.str(),
			"#PREPARE_LOAD_INPUT_ALL",PREPARE_LOAD_INPUT_ALL.str(),
			"#REDUCE_INIT_ALL",REDUCE_INIT_ALL.str(),
			"#REDUCE_INIT_SHARED", REDUCE_INIT_SHARED.str(),
			"#LOAD_INPUT_ALL",LOAD_INPUT_ALL.str(),
			"#LOAD_REDUCE_ALL",LOAD_REDUCE_ALL.str(),
			"#SAVE_REDUCE_ALL",SAVE_REDUCE_ALL.str(),
			"#LOAD_REDUCED_SAVE_GLOBAL_ALL",LOAD_REDUCED_SAVE_GLOBAL_ALL.str(),
			"#REDUCE", core::format_code(reduce),
			"#CALC", core::format_code(compute_code));
	}
	return mPointwisePrograms[k];
}

	
PerDeviceProgramCache& PerDeviceProgramCache::instance()
{
	static PerDeviceProgramCache cache;
	return cache;
}

tart::program_ptr PerDeviceProgramCache::getPointwiseBroadcastReduceOperation(
	const tart::device_ptr& device,
	std::vector<TensorSpecs>& xs,
	std::vector<TensorSpecs>& ys,
	int weights_count,
	const tart::DType& weights_type,
	const std::string& compute_code,
	const std::string& reduce_init,
	const std::string& reduce)
{
	return getPointwiseCache(device).getPointwiseBroadcastReduceOperation(xs, ys, weights_count, weights_type, compute_code, reduce_init, reduce);
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

PointwiseCache& PerDeviceProgramCache::getPointwiseCache(const tart::device_ptr& device)
{
	std::uintptr_t key = (std::uintptr_t)device.get();
	if (mPointwiseCaches.find(key) == mPointwiseCaches.end())
	{
		mPointwiseCaches[key] = std::make_unique<PointwiseCache>(device);
	}
	return *mPointwiseCaches[key];
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
		bool use_io_type = (dtypes[0] == dtypes[1]);
		tart::DType dt0 = dtypes[0];
		tart::DType dt1 = dtypes[1];
		
		// avoid compilation errors
		if (dt1.isFloatingPoint())
		{
			mCopyStridedProgram = gpu::Cache::instance().get_program(device, "copy_strided", "dtype_src", dt0.glsl(), "dtype_tgt", dt1.glsl() );
		}
		if (dt0.isFloatingPoint())
		{
			mNullLossBwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_bwd", "dtype", dt0.glsl(), "itype", dt1.glsl());
			mNullLossFwdProgram = gpu::Cache::instance().get_program(device, "nll_loss_fwd", "dtype", dt0.glsl(), "itype", dt1.glsl());
		}
		
		mPointwiseUnaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-unary-unary", "typeof_x0", dt0.glsl(), "typeof_y0", dt1.glsl());
	}
	else if(dtypes.size() == 3)
	{
		// Disable, as it currently does not compile
		// mGemm2Program = gpu::Cache::instance().get_program(device, "gemm2", "A_TYPE", dtypes[0].glsl(), "B_TYPE", dtypes[1].glsl(), "D_TYPE", dtypes[2].glsl());
		
		mPointwiseBinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-binary-unary",
			"typeof_x0", dtypes[0].glsl(),
			"typeof_x1", dtypes[1].glsl(),
			"typeof_y0", dtypes[2].glsl());
		
		mPointwiseUnaryBinaryProgram = gpu::Cache::instance().get_program(device, "pointwise-unary-binary",
			"typeof_x0", dtypes[0].glsl(),
			"typeof_y0", dtypes[1].glsl(),
			"typeof_y1", dtypes[2].glsl());
	}
	else if (dtypes.size() == 4)
	{
		mPointwiseTrinaryUnaryProgram = gpu::Cache::instance().get_program(device, "pointwise-trinary-unary",
			"typeof_x0", dtypes[0].glsl(),
			"typeof_x1", dtypes[1].glsl(),
			"typeof_x2", dtypes[2].glsl(),
			"typeof_y0", dtypes[3].glsl());
	}
}

} // namespace gpu

} // namespace dlprim
