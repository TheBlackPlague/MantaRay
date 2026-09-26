//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
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

#if defined(_MSC_VER)
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#endif
