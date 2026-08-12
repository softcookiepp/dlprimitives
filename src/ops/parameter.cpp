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
    void Parameter::copy_and_scale(Tensor &tgt,Tensor &src,float accum,ExecutionContext const &q)
    {
        DLPRIM_CHECK(tgt.specs() == src.specs());

        {
            if(accum == 0)
                core::pointwise_operation({src},{tgt},{},"y0=x0;");
            else
                core::pointwise_operation({src,tgt},{tgt},{accum},"y0=x0+w0*x1;");
        }
    }
}
