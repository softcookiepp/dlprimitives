///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/parameter.hpp>
#include <my_cblas.hpp>
#include <dlprim/core/pointwise.hpp>
namespace dlprim
{
    void Parameter::copy_and_scale(Tensor &tgt,Tensor &src,float accum)
    {
        DLPRIM_CHECK(tgt.specs() == src.specs());

        {
            if(accum == 0)
				#if 1
					core::pointwiseOpStrided({src}, {tgt}, {}, core::PointwiseOp::eIdentity);
				#else
					core::pointwise_operation({src},{tgt},{},"y0=x0;");
				#endif
            else
                core::pointwise_operation({src,tgt},{tgt},{accum},"y0=x0+w0*x1;");
        }
    }
}
