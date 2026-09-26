//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_CLIPPEDRELU_H
#define MANTARAY_CLIPPEDRELU_H

#include "../../Common/Integral.h"

namespace MantaRay
{

    template<i32 Min, i32 Max>
    struct ClippedReLU
    {

        constexpr static i32 Minimum = Min;
        constexpr static i32 Maximum = Max;

    };

}

#endif
