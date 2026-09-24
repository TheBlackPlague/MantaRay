//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_SQUAREDCLIPPEDRELU_H
#define MANTARAY_SQUAREDCLIPPEDRELU_H

#include "../../Common/Integral.h"

namespace MantaRay
{

    template<i32 Min = 0, i32 Max = 1>
    struct SquaredClippedReLU
    {

        static constexpr i32 Minimum = Min;
        static constexpr i32 Maximum = Max;

    };

}

#endif
