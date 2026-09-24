//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
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
