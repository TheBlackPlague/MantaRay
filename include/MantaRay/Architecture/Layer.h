//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#pragma once

#include "Activation/Identity.h"
#include "Transform/Affine.h"

namespace MantaRay
{

    template<std::size_t I, std::size_t O, typename A = Identity, typename T = Affine>
    struct Layer
    {

        static constexpr std::size_t  InputSize = I;
        static constexpr std::size_t OutputSize = O;

        using Activation = A;
        using Transform = T;

    };

}
