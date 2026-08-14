///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include "tart.hpp"

#if defined(__WIN32) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__)
#   define  DLPRIM_WINDOWS
#	if defined(DLL_EXPORT)
#		if defined(DLPRIM_SOURCE)
#			define DLPRIM_API __declspec(dllexport)
#		else
#			define DLPRIM_API __declspec(dllimport)
#		endif
#	else
#		define DLPRIM_API
#	endif
#else // ELF BINARIES
#	define DLPRIM_API  __attribute__((visibility("default")))
#endif     


///
/// Mane namespace
///
namespace dlprim {
    namespace json { class value; }

    ///
    /// Base dlprim excetion
    ///
    class Error : public std::runtime_error {
    public:
        Error(std::string const &v) : std::runtime_error(v) {}
    };

    ///
    /// Thrown if some stuff is not implemented yet
    ///
    class NotImplementedError : public Error {
    public:
        NotImplementedError(std::string const &v) : Error(v) {}
    };

    ///
    /// Thrown in case of invalid parameters
    ///
    class ValidationError : public Error {
    public:
        ValidationError(std::string const &v) : Error(v) {}
    };

    ///
    /// Thrown if OpenCL kernel compilation failed.
    ///
    class BuildError : public Error {
    public:
        BuildError(std::string const &msg,std::string const &log) : Error(msg), log_(log) {}
        /// get full build log
        std::string const &log() const 
        {
            return log_;
        }
    private:
        std::string log_;
    };

    #define DLPRIM_CHECK(x) \
    do { if(!(x)) throw ValidationError(std::string("Failed " #x " at " __FILE__ ":") + std::to_string(__LINE__) ); } while(0)
    
	inline std::string data_type_to_string(const tart::DType& dt)
	{
		return dt.glsl();
	}
    enum DataTypeLimit {
        dt_min_val,
        dt_max_val,
    };

    inline std::string data_type_to_opencl_numeric_limit(const tart::DType& dt, DataTypeLimit lmt)
    {
		#if 0
		#else
			// TODO: move this to tart as well
			if(dt.isFloatingPoint())
			{
				std::string prefix;
				if(dt == tart::dtypes::float32 || dt == tart::dtypes::float16) prefix="FLT";
				else if (dt == tart::dtypes::float64) prefix="DBL";
				else if (dt == tart::dtypes::float16) prefix="HALF";
				else throw ValidationError("Unsupported type");
				switch(lmt)
				{
					case dt_min_val: return "(-" + prefix + "_MAX)";
					case dt_max_val: return prefix + "_MAX";
				};
			}
			else
			{
				if (lmt == dt_min_val
					&& (dt == tart::dtypes::uint64 || dt == tart::dtypes::uint32 || dt == tart::dtypes::uint16 || dt == tart::dtypes::uint8))
						return 0;
				std::string prefix;
					 if (dt == tart::dtypes::int64)  prefix = "LONG";
				else if (dt == tart::dtypes::uint64) prefix = "ULONG";

				else if (dt == tart::dtypes::int32)  prefix = "INT";
				else if (dt == tart::dtypes::uint32) prefix = "UINT";

				else if (dt == tart::dtypes::int16)  prefix = "SHRT";
				else if (dt == tart::dtypes::uint16) prefix = "USHRT";

				else if (dt == tart::dtypes::int8)   prefix = "CHAR";
				else if (dt == tart::dtypes::uint8)  prefix = "UCHAR";
				else throw NotImplementedError("Unsupported data type");
				switch(lmt)
				{
					case dt_min_val: return prefix + "_MIN";
					case dt_max_val: return prefix + "_MAX";
				};
			}
			throw NotImplementedError("Unsupported data type");
		#endif
    }

    /// Maximal number of dimensions in tensor
    static constexpr int max_tensor_dim = 8;

    /// internal flag
	constexpr int forward_data = 1;
    /// internal flag
	constexpr int backward_data = 2;
    /// internal flag
	constexpr int backward_param = 3;

	///
    /// Parameterless Activations that can be embedded to general kernels like inner product or convolution
    ///
    enum class StandardActivations : int {
		identity = 0,
		relu = 1,
        tanh = 2,
        sigmoid = 3,
        relu6 = 4,
	};
    
    std::string activation_equation(StandardActivations act,std::string const &variable); 
    std::string activation_backward_equation(StandardActivations act,std::string const &dy,std::string const &y);


    ///
    /// Operation mode of layers - inference of training
    ///
    enum class CalculationsMode {
        train,
        predict
    };
    

    /// 
    /// internal GEMM mode
    ///
    enum class GemmOpMode {
        forward = 1,
        backward_filter = 2,
        backward_data = 3
    };
	
    ///
    /// Convolution settings
    ///
    struct Convolution2DConfigBase {
		int channels_in = -1;
		int channels_out = -1;
		int kernel[2] = {1,1};
		int stride[2] = {1,1};
		int dilate[2] = {1,1};
		int pad[2] = {0,0};
		int groups = 1;
    };
    ///
    /// Interpolation methods
    ///
    enum class InterpolateType {
        nearest  = 0,
        nearest_exact  = 1,
        bilinear = 2
    };



}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

