#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <sstream>

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

tart::program_ptr 
	PointwiseCache::getPointwiseOperation(std::vector<Tensor>& xs,
		std::vector<Tensor>& ys, std::vector<double> ws, const std::string& code)
{
	tart::device_ptr device = mDevice.lock();
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
	std::get<3>(k) = code;
	
	if (mPointwisePrograms.find(k) == mPointwisePrograms.end())
	{
		Shape ref;
		tart::DType ref_type = tart::dtypes::float32;
		DLPRIM_CHECK(xs.size() + ys.size() > 0);
		if(xs.empty()) {
			ref = ys[0].shape();
			ref_type = ys[0].tDtype();
		}
		else {
			ref = xs[0].shape();
			ref_type = xs[0].tDtype();
		}

		for(size_t i=0;i<xs.size();i++)
		{
			DLPRIM_CHECK(ref == xs[i].shape());
			DLPRIM_CHECK(ref_type == xs[i].tDtype());
		}
		for(size_t i=0;i<ys.size();i++)
		{
			DLPRIM_CHECK(ref == ys[i].shape());
			DLPRIM_CHECK(ref_type == ys[i].tDtype());
		}
		std::ostringstream params,loads,saves;
		size_t bindingIndex = 0;
		std::stringstream bufferDefs;
		for (size_t i = 0; i < xs.size(); i += 1)
		{
			// wait. we can't have macros within macros. this will not work.
			
			//params << "\n#if USE_BDA\n dtype_addr_ro px" << i << ";\n#endif\n"
			//	<< "uint px" << i << "_offset;\n";
			params << "uint px" << i << "_offset; ";
			bufferDefs << "	layout(binding = " << bindingIndex << ", std430) readonly buffer px"
				<< i << "_buf { dtype px" << i << "[]; }; ";
			loads << "	dtype x" << i << " = px" << i << "[index + px" << i << "_offset]; ";
			bindingIndex += 1;
		}
		for (size_t i = 0; i < ys.size(); i += 1)
		{
			//params << "\n#if USE_BDA\ndtype_addr_rw py" << i << "; #endif\n"
			//	<< "uint py" << i << "_offset;\n";
			params << "uint py" << i << "_offset; ";
			bufferDefs << "	layout(binding = " << bindingIndex << ", std430) buffer py"
				<< i << "_buf { dtype py" << i << "[]; }; ";
			loads << "dtype y" << i << "; ";
			saves << "py" << i << "[index] = y" << i << "; "; // no offset?
			bindingIndex += 1;
		}
		
		std::string param_dtype = ref_type.glsl();
		for (size_t i = 0; i < ws.size(); i += 1)
		{
			params << param_dtype << " w" << i << "; ";
		}
		
		// what is this for? we will find out later I guess
		std::ostringstream code_fixed;
		for(size_t i=0;i<code.size();i++)
			if(code[i]=='\n')
				code_fixed << "\\\n";
			else
				code_fixed << code[i];
		mPointwisePrograms[k] = gpu::Cache::instance().get_program(device, "pointwise",
																		   "dtype", ref_type.glsl(),
																		   "#BUFFER_DEFS", bufferDefs.str(),
																		   "#PARAMS",params.str(),
																		   "#LOADS",loads.str(),
																		   "#SAVES",saves.str(),
																		   "#CALC",code_fixed.str());
	}
	return mPointwisePrograms[k];
}

tart::program_ptr PointwiseCache::getPointwiseBroadcastOperation(
		const std::vector<Tensor>& xs,
		const std::vector<Tensor>& ys,
		const std::vector<double>& ws,
		const std::vector<tart::DType>& dts,
		const std::string &code,
		const bool shrinkDims)
{
	tart::device_ptr device = mDevice.lock();
	tart::program_ptr prog = nullptr;
	if(!prog)
	{
		tart::DType target_type = ys[0].tDtype();
		
		size_t bindingIndex = 0;
		std::stringstream bufferDefs;
		std::stringstream typeDefs;
        std::ostringstream params,loads,saves;
        for(size_t i=0;i<xs.size();i++) {
            std::string type = xs[i].tDtype().glsl();
            params << "uint px" << i << "_offset; Shape strides" << i << "; ";
			bufferDefs << "	layout(binding = " << bindingIndex << ", std430) readonly buffer px"
				<< i << "_buf { " << type << " px" << i << "[]; }; ";
			bindingIndex += 1;
            loads<<type << " x"<<i<<"=px"<<i<<"[get_offset(index,strides" << i << ",px"<<i<<"_offset)]; ";
            typeDefs << "#define typeof_x" << i << " " << type << "\n";
        }
        for(size_t i=0;i<ys.size();i++) {
            std::string type = ys[i].tDtype().glsl();
            params << "uint py" << i << "_offset; ";
			bufferDefs << "	layout(binding = " << bindingIndex << ", std430) buffer py"
				<< i << "_buf { " << type << " py" << i << "[]; }; ";
			bindingIndex += 1;
            loads<<type << " y"<<i<<";\\\n";
            saves<<"py"<<i<<"[get_direct_offset(index,limit,py"<<i<<"_offset)]=y"<<i<<";\n";
            typeDefs << "#define typeof_y" << i << " " << type << "\n";
        }
        typeDefs << "#define target_type " << target_type.glsl() << "\n";

        for(size_t i=0;i<ws.size();i++) {
            std::string type = dts[i].glsl();
            params << type << " w" << i << "; ";
            typeDefs << "#define typeof_w" << i << " " << type << "\n";
        }

        loads << '\n';
        saves <<'\n';
		prog = gpu::Cache::instance().get_program(device,  "pointwise_broadcast",
			"$TYPEDEFS", typeDefs.str(),
			"#BUFFER_DEFS", bufferDefs.str(),
			"#PARAMS", params.str(),
			"#LOADS", loads.str(),
			"#SAVES", saves.str(),
			"#CALC", core::format_code(code));
	}
	return prog;
}
	
PerDeviceProgramCache& PerDeviceProgramCache::instance()
{
	static PerDeviceProgramCache cache;
	return cache;
}

tart::program_ptr PerDeviceProgramCache::getPointwiseOperation(const tart::device_ptr& device, std::vector<Tensor>& xs,
	std::vector<Tensor>& ys, std::vector<double> ws, const std::string& code)
{
	return getPointwiseCache(device).getPointwiseOperation(xs, ys, ws, code);
}

tart::program_ptr PerDeviceProgramCache::getPointwiseBroadcastOperation(
	const tart::device_ptr& device,
	const std::vector<Tensor>& xs,
	const std::vector<Tensor>& ys,
	const std::vector<double>& ws,
	const std::vector<tart::DType>& dts,
	const std::string &code,
	const bool shrinkDims)
{
	//return getPointwiseCache(device).getPointwiseBroadcastOperation(xs, ys, ws, code);
	throw std::runtime_error("not implemented");
	return nullptr;
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
