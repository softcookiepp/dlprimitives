#pragma once
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>


namespace dlprim
{

namespace gpu
{

enum class SoftmaxEpilogue
{
	eForward = 1,
	eBackward = 2,
	eLogForward = 3,
	eLogBackward = 4
};

void spatial_softmax(
	const ExecutionContext& e,
	const DataType dtype,
	const SoftmaxEpilogue epilogue,
	const tart::buffer_ptr& output_buffer,
	const uint32_t output_offset,
	const tart::buffer_ptr& input_buffer,
	const uint32_t input_offset,
	const uint32_t outer_size,
	const uint32_t dim_size,
	const uint32_t inner_size,
	const bool half_to_float,
	const tart::command_sequence_ptr& sequence = nullptr);

} // namespace gpu

} // namespace dlprim
