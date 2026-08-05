#include <dlprim/gpu/softmax.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <iostream>
#include <algorithm>

namespace dlprim
{

namespace gpu
{

using tart::dim3;
	
const uint32_t max_threads = 1024;

inline dim3 SpatialSoftMax_getBlockSize(
	uint32_t dim_size, uint32_t inner_size)
{
	uint32_t inner_threads = inner_size;
	inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
	uint32_t dim_threads = 1;
	if (inner_threads <= 64 && dim_size >= 64)
	{
		while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
			dim_threads *= 2;
		dim_threads /= 2;
	}
	return dim3(dim_threads, inner_threads);
}

inline dim3 SpatialSoftMax_getGridSize(
		dim3 block, dim3 max_active_blocks,
		uint32_t outer_size, uint32_t inner_size)
{
	// First, tile as many blocks as we can over the y axis
	uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
	if (inner_blocks > max_active_blocks.y)
		inner_blocks = max_active_blocks.y;
	// Fill the x axis with as many blocks as we can fit (a little more is ok too)
	uint32_t outer_blocks = (max_active_blocks.x + inner_blocks - 1) / inner_blocks;
	if (outer_blocks > outer_size)
		outer_blocks = outer_size;
	return dim3(outer_blocks, inner_blocks);
}

void SpatialSoftMax_getLaunchSizes(
		//Kernel k,
		const tart::device_ptr& device,
		uint32_t outer_size, uint32_t dim_size, uint32_t inner_size,
		dim3& grid, dim3& block, uint32_t& smem_size)
{
	block = SpatialSoftMax_getBlockSize(dim_size, inner_size);
	uint32_t block_threads = block.x * block.y;
	smem_size = block.x == 1 ? 0 : block_threads; // shared memory size in Vulkan GLSL is in elements of whatever type you are using, not bytes
	dim3 max_active_blocks;
	max_active_blocks.x = device->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
	max_active_blocks.y = device->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
	max_active_blocks.z = device->getMetadata().physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
	grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, inner_size);
}

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
	const bool half_to_float)
{
	Context ctx(e);
	
	uint32_t smem_size;
	dim3 grid, block;
	
	if (!half_to_float)
	{
		SpatialSoftMax_getLaunchSizes(
				e.queue(),
				outer_size, dim_size, inner_size,
				grid, block, smem_size);
		tart::program_ptr prg = gpu::PerDeviceProgramCache::instance().spatial_softmax(ctx.device(), dtype);
		tart::kernel_ptr k = prg->getKernel("softmax_forward");
		
		int p = 0;
		k->setArg(p++, output_buffer);
		k->setArg(p++, output_offset);
		k->setArg(p++, input_buffer);
		k->setArg(p++, input_offset);
		k->setArg(p++, outer_size);
		k->setArg(p++, dim_size);
		k->setArg(p++, inner_size);
		
		std::vector<uint32_t> g = grid.toVector();
		std::vector<uint32_t> spec = block.toVector();
		spec.push_back(smem_size);
		k->enqueue(g, spec);
	}
	else
	{
		throw std::runtime_error("half_to_float not implemented");
#if 0
		SpatialSoftMax_getLaunchSizes<accscalar_t>(
				&cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, index_t, Epilogue>,
				outer_size, dim_size, inner_size,
				grid, block, smem_size);
		cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, index_t, Epilogue>
			<<<grid, block, smem_size, stream>>>(
			output.mutable_data_ptr<accscalar_t>(), input.const_data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
#endif
	}
}

void spatial_softmax_backward(
	const ExecutionContext& e,
	const DataType dtype,
	const SoftmaxEpilogue epilogue,
	const tart::buffer_ptr& gI,
	uint32_t gI_offset,
	const tart::buffer_ptr& output,
	uint32_t output_offset,
	const tart::buffer_ptr& grad,
	uint32_t grad_offset,
	uint32_t outer_size,
	uint32_t dim_size,
	uint32_t inner_size,
	bool half_to_float)
{
	Context ctx(e);
	
	uint32_t smem_size;
	dim3 grid, block;

	if (!half_to_float)
	{
		SpatialSoftMax_getLaunchSizes(e.queue(), outer_size, dim_size, inner_size, grid, block, smem_size);
		tart::program_ptr prg = gpu::PerDeviceProgramCache::instance().spatial_softmax(ctx.device(), dtype);
		tart::kernel_ptr k = prg->getKernel("softmax_backward");
		
		int p = 0;
		k->setArg(p++, gI);
		k->setArg(p++, gI_offset);
		k->setArg(p++, output);
		k->setArg(p++, output_offset);
		k->setArg(p++, grad);
		k->setArg(p++, grad_offset);
		k->setArg(p++, outer_size);
		k->setArg(p++, dim_size);
		k->setArg(p++, inner_size);
		
		std::vector<uint32_t> g = grid.toVector();
		std::vector<uint32_t> spec = block.toVector();
		spec.push_back(smem_size);
		k->enqueue(g, spec);
	}
	else
	{
#if 1
		throw std::runtime_error("half_to_float not implemented");
#else
		SpatialSoftMax_getLaunchSizes<accscalar_t>(
				&cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
				outer_size, dim_size, inner_size,
				grid, block, smem_size);

		cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>
			<<<grid, block, smem_size, stream>>>(
				gI.mutable_data_ptr<scalar_t>(), output.const_data_ptr<accscalar_t>(), grad.const_data_ptr<accscalar_t>(),
				outer_size, dim_size, inner_size
		);
#endif
	}
}

} //namespace gpu

} // namespace dlprim
