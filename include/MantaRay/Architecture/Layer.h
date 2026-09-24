//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_LAYER_H
#define MANTARAY_LAYER_H

#include "Activation/Identity.h"
#include "Transform/Affine.h"

#include "../Common/Integral.h"

namespace MantaRay
{

    template<s00 I, s00 O, typename A = Identity, typename T = Affine>
    struct Layer
    {

        static constexpr s00  InputSize = I;
        static constexpr s00 OutputSize = O;

        using Activation = A;
        using Transform = T;

    };

}

#endif
