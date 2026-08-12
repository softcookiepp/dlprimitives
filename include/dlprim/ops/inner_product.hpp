///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {
    namespace gpu { class GEMM; }
    namespace core { class IPForward; class IPBackwardData; class IPBackwardFilter; }
    class BWBias;

	struct InnerProductConfig {
        int inputs = -1;
		int outputs = -1;
		bool bias = true;
		StandardActivations activation = StandardActivations::identity;
	};


	class InnerProduct : public Operator {
	public:

        InnerProduct(Context &ctx,InnerProductConfig const &cfg);
        virtual ~InnerProduct();

        virtual char const *operator_type() const
        {
            return "InnerProduct";
        }
        void initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e);
		
        virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);

        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);


	protected:

		InnerProductConfig config_;
        DataType dtype_;
        std::unique_ptr<core::IPForward> ip_;
        std::unique_ptr<core::IPBackwardData> bwd_ip_;
        std::unique_ptr<core::IPBackwardFilter> bwd_weights_ip_;
        std::unique_ptr<Operator>  activation_;
        std::unique_ptr<BWBias> bwd_bias_;
	};
	
}
