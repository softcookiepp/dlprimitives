#include <dlprim/gpu/im2col.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/gpu/tiered_cache.hpp>
#include <cmath>

namespace dlprim
{
namespace gpu
{

void gemm2Routine(
	uint32_t M, uint32_t N, uint32_t K,
	tart::buffer_ptr& a,
	const tart::DType& aType,
	uint32_t offset_a,
	uint32_t lda,
	tart::buffer_ptr& b,
	const tart::DType& bType,
	uint32_t offset_b,
	uint32_t ldb,
	tart::buffer_ptr& c,
	const tart::DType& cType,
	uint32_t offset_c,
	uint32_t ldc)
{
	tart::device_ptr device = a->getDevice();
	tart::program_ptr prg = PerDeviceProgramCache::instance().gemm2(device, aType, bType, cType);
	
	uint32_t baseWorkGroupZ = 0; // The base work group of each batch?
	uint32_t kSplit = 0; // ?
	uint32_t wgX = static_cast<uint32_t>(std::ceil(static_cast<float>(M)/64.0));
	std::vector<uint32_t> push = {
		M, N, K, lda, ldb, ldc,
		0, 0, 0, // batch strides (not implemented yet)
		baseWorkGroupZ,
		1, // num batches (just 1 for now)
		K,
		ne02,
		ne12,
		broadcast2, broadcast3,
		padded_n
	};
}

} // namespace gpu
} // namespace dlprim
