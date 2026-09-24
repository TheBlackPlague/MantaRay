//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINER_H
#define MANTARAY_CONTAINER_H

#include <array>

#include "Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N, s00... Ns>
    struct IArray { using Internal = std::array<typename IArray<T, Ns...>::Internal, N>; };

    template<QuantizedInteger T, s00 N>
    struct IArray<T, N> { using Internal = std::array<T, N>; };

    template<QuantizedInteger T, s00 N>
    using Array = std::array<T, N>;

    template<QuantizedInteger T, s00 N, s00... Ns>
    using NArray = IArray<T, N, Ns...>::Internal;

}

#endif
