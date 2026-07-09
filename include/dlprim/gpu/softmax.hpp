#pragma once


namespace dlprim
{

namespace gpu
{

enum class SoftmaxEpilogue
{
	eForward = 1,
	eBackward = 2
	eLogForward = 3,
	eLogBackward = 4
};



} // namespace gpu

} // namespace dlprim
