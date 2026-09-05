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
		eBceBwd = 69
	};
    
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
    
    ///
    /// Similar to pointwise_operation but xs are broadcasted numpy style. ys must much broadcasted shape, weights are considered
    /// of ys[0].dtype()
    ///
    #if 0
    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double>  weights,
                                        std::string const &code);
	#endif

    ///
    /// Similar to pointwise_operation but xs are broadcasted numpy style. ys must much broadcasted shape
    ///
    #if 0
    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double>  weights,
                                        const std::vector<tart::DType>& weights_types,
                                        std::string const &code,
                                        bool shrink_dims=true);
	#endif
    ///
    /// Perform pointwise operation with both boradcasting and reduction
    ///
    /// Calculation is performed over a shape that xs and ys tensors are boradcaasted to.
    ///
    /// For example xs have shapes: (64,10,5) and (64,10,1) and ys has shape (10,1) they all
    /// broadcast to 64,10,5 and reduction is performed over dimentsions 0 and 2
    ///
    /// All ys tensors need to have same shape and be boradcastable to total shape
    ///
    /// Optional parameters can be provided that avalible in code as w0... wN, Final ys are computed as `ys[i] = alpha[i] * reduced_result + beta[i] * ys[i]`
    ///
    class PointwiseOperationBroadcastReduce {
    public:
        
        virtual ~PointwiseOperationBroadcastReduce() {}
        ///
        /// Get size of workspace in bytes needed
        ///
        virtual size_t workspace() = 0;
        ///
        /// Perform coputations
        ///
        /// \param xs - vector of input tensor
        /// \param ys - vector of output tenors
        //  \param parameters - the weight paramerters, size should match weights_count
        /// \param alpha - scale for ys, must match size of ys
        /// \param beta - scale for summation of previous ys, must match size of ys
        ///
        ///
        virtual void enqueue(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             Tensor &workspace,
                             std::vector<double> parameters,
                             std::vector<double> alpha,
                             std::vector<double> beta) = 0;

        ///
        /// Create objects:
        ///
        /// \param xs - vector of input tensor specs - such tensors are expected to be given to enqueue
        /// \param ys - vector of output tenorr specs - such tensors are expectred to be give to enqueue
        //  \param weights_count - size of parameters vector in enqueue
        /// \param weights_type - type of weights parameters as provided
        ///
        /// \param compute_code - OpenCL code to compute values. You can use x0, x1, ... xN as input values for each x for xs
        /// y0,.., yN for each output and w0,...,wN for each weight. For example "y0 = x0 + w0 * x1;"
        ///
        /// \param reduce_init - initalization of reduction variables `reduce_yN` for example "reduce_y0 = 0;" or "reduce_y0=-FLT_MAX;"
        /// \param reduce - code for sum reduction "reduce_y0 += y0" or max reduction "reduce_y0 = max(reduce_y0,y0)"
        ///
        static std::unique_ptr<PointwiseOperationBroadcastReduce> create(
                        const tart::device_ptr& device,
                        std::vector<TensorSpecs> xs,
                        std::vector<TensorSpecs> ys,
                        int weights_count,
                        const tart::DType& weights_type,
                        std::string const &compute_code,
                        std::string const &reduce_init,
                        std::string const &reduce);

    };

} // core
} // dlprim
