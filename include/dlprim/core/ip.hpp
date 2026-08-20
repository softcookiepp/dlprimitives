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
#include <optional>
namespace dlprim {
///
/// All Basic Operations on GPU
///
namespace core {
    ///
    /// Configuration of InnerProduct layer 
    ///
    struct IPSettings
    {
        int inputs = -1;   /// number of input features 
        int outputs = -1;  /// output features
        int optimal_batch_size = -1;  /// Expected batch size the network is used with
        tart::DType dtype = tart::dtypes::float32;
    };
    
    ///
    /// Perform InnerProduct/FullyConnected/Dense forward calulations, allow fusing bias and activation
    /// into same GPU kernel (not quite ported yet)
    /// 
    void ipForward(Tensor& x, Tensor& w, Tensor& y, StandardActivations activations  = StandardActivations::identity, std::optional<Tensor> bias = {});

    ///
    /// Perform InnerProduct/FullyConnected/Dense backward data calculations
    /// 
    void ipBackwardData(Tensor& dx, Tensor& M, Tensor& dy, float factor);

    ///
    /// Perform InnerProduct/FullyConnected/Dense backward filter calcilations
    ///
    class IPBackwardFilter {
    public:
        virtual ~IPBackwardFilter() {}
        virtual void enqueue(Tensor &x,Tensor &dw,Tensor &dy,float factor) = 0;
        ///
        /// Create optimal object for innter product calculation
        ///
        /// config - IP Settings,
        static std::unique_ptr<IPBackwardFilter> create(const tart::device_ptr& device,IPSettings const &config);
    };

} // core
} // dlprim
