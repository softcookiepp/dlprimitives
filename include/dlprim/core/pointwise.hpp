///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    ///
    /// Bind a parameter to kernet casting it to apropriate opencl type dt
    ///
    void bind_as_dtype(tart::kernel_ptr k,int &p,double value, const tart::DType& dt);
    
    std::string format_code(std::string const &code);
    
    enum class PointwiseOp
    {
		// unary x, unary y
		eIdentity = 0,
		eFill = 1,
		eAdd = 2,
		eSub = 3,
		eMul = 4,
		eDiv = 5,
		eAxpy = 6,
		eScale = 7,
		eAddScalar = 8,
		eSubScalar = 9,
		eDivScalar = 10,
		eRsubScalar = 11,
		eRdivScalar = 12,
		ePow = 13,
		eAxpb = 14,
		eAxpby = 15,
		eHardtanh = 16,
		eHardtanhBwd = 17,
		eAbs = 18,
		eAtan = 19,
		eLog = 20,
		eSqrt = 21,
		eExp = 22,
		eSgn = 23,
		eHardswish = 24,
		eHardsigmoid = 25,
		eHardsigmoidBwd = 26,
		eHardswishBwd = 27,
		eSilu = 28,
		eSiluBwd = 29,
		eLeakyRelu = 30,
		eLeakyReluBwd = 31,
		eBitwiseNot = 32,
		eLogicalNot = 33,
		eClamp = 34,
		eCeil = 35,
		eGelu = 36,
		eGeluApproximate = 37,
		eGeluBwd = 38,
		eGeluApproximateBwd = 39,
		eLogSigmoid = 40,
		eLogit = 41,
		eArange = 42,
		eLogSigmoidBwd = 43,
		eThreshold = 44,
		eThresholdBwd = 45,
		eDropout = 46,
		eRound = 47,
		eNeg = 48,
		eRecip = 49,
		eAddcmul = 50,
		eCmpGt = 51,
		eCmpLt = 52,
		eCmpGe = 53,
		eCmpLe = 54,
		eCmpEq = 55,
		eCmpNe = 56,
		eLerp = 57,
		eAddcdiv = 58,
		eMseBwd = 59,
		eFma = 60,
		eTransformBiasRescaleQKV = 61,
		eBitwiseAnd = 62,
		eBitwiseOr = 63,
		eBitwiseXor = 64,
		eLogicalAnd = 65,
		eLogicalOr = 66,
		eMax = 67,
		eMin = 68,
		eBceBwd = 69,
		eBcdFwdWeightless = 70,
		eBcdFwd = 71,
		eMse = 72,
		eLayerGroupNormBwd = 73,
		eArgmaxReduce = 74,
		eArgmaxInit = 75
	};
	
	std::vector<int> getReduceDims(dlprim::Shape ref, std::vector<int> dim);
    
    // Pointwise operation, but without code generation requirement.
    // Instead, chosen routine is chosen by PointwiseOp value provided
    void pointwiseOpStrided(std::vector<Tensor> xs,
			std::vector<Tensor> ys,
			std::vector<float> ws,
			const PointwiseOp op,
			const tart::DType& acctype = tart::dtypes::float32,
			const tart::DType& iacctype = tart::dtypes::int32);
	
	// Just like the above, but with auto-broadcasting.
	void pointwiseOpBroadcastStrided(std::vector<Tensor> xs,
			std::vector<Tensor> ys,
			std::vector<float> ws,
			const PointwiseOp op,
			const tart::DType& acctype = tart::dtypes::float32,
			const tart::DType& iacctype = tart::dtypes::int32);
    
	// Perform reduce operations on either contiguous or non-contiguous tensors
    void pointwiseOpBroadcastReduceStrided(std::vector<Tensor> xs, std::vector<Tensor> ys, std::vector<float> ws,
		std::vector<int> reduceDims,
		PointwiseOp calcOp, PointwiseOp reduceOp, std::vector<float> yInitValues);
    
    void bind_shape(tart::kernel_ptr k, int &p,Shape const &s);

} // core
} // dlprim
