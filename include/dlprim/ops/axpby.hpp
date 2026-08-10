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
	class AXPBY {
	public:
		AXPBY(Context &ctx, const tart::DType& dtype = tart::dtypes::float32);
		~AXPBY();
		void apply(float a,Tensor &x,float b,Tensor &y,Tensor &z,ExecutionContext const &e);
	private:
		Context ctx_;
		tart::kernel_ptr kernel_;
	};
} // namespace
