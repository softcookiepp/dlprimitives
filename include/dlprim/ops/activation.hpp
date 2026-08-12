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
    struct ActivationConfig {
        StandardActivations activation = StandardActivations::identity;
    };


   
    class Activation : public Operator
    {
    public:
        
        Activation(Context &ctx,ActivationConfig config = ActivationConfig());
        virtual ~Activation();
        
        virtual char const *operator_type() const
        {
            return "Activation";
        }

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

        static std::unique_ptr<Activation> get_bwd_op(Context &ctx,StandardActivations act,TensorSpecs spec);
        
    private:
        ActivationConfig config_;
        DataType dtype_;
    };
} // namespace
 
