//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CLIPPEDRELU_H
#define MANTARAY_CLIPPEDRELU_H

#include "../../Common/Integral.h"

namespace MantaRay
{

    template<i32 Min, i32 Max>
    struct ClippedReLU
    {

        static constexpr i32 Minimum = Min;
        static constexpr i32 Maximum = Max;

    };

}

#endif
