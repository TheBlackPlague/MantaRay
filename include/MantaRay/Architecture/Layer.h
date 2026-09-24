//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
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
