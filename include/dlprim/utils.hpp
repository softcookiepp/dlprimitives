#if VULKAN_API
namespace dlprim
{

size_t CeilDiv(const size_t x, const size_t y) { return 1 + ((x - 1) / y); }
size_t Ceil(const size_t x, const size_t y) { return CeilDiv(x, y) * y; }

} // end namespace dlprim
#endif
