#include <dlprim/ops/bwd_bias.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/bias.hpp>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
	BWBias::~BWBias() {}

	BWBias::BWBias(Context &ctx,Shape const &sp,DataType dt) 
	{
		impl_ = std::move(core::BiasBackwardFilter::create(ctx,sp,dt));
	}

	size_t BWBias::workspace() const
	{
		if(impl_.get())
			return impl_->workspace();
		return 0;
	}
	void BWBias::backward(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e)
	{
		DLPRIM_CHECK(dy.shape().size() >= 2);
		DLPRIM_CHECK(dw.shape().size() == 1);
		DLPRIM_CHECK(dw.shape().total_size() == dy.shape()[1]);
		impl_->enqueue(dy,dw,ws,beta,e);
	}
}
