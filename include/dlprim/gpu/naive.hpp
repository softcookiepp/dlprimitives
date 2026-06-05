#pragma once
#if VULKAN_API
#include <dlprim/core/util.hpp>
#include <optional>

namespace dlprim
{

namespace gpu
{

// all must be made contiguous prior to this
Tensor convolution_from_ggml_raw(const Tensor& input,
	const Tensor& weight,
	const std::optional<Tensor>& bias,
	const std::vector<int>& stride,
	const std::vector<int>& padding,
	const std::vector<int>& dilation,
	bool transposed,
	const std::vector<int>& output_padding,
	int64_t groups);

}

} // dlprim

#endif
