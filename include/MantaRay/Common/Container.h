//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_CONTAINER_H
#define MANTARAY_COMMON_CONTAINER_H

#include <array>

#include "Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N>
    using Array = std::array<T, N>;

    namespace Detail
    {

        template<typename T, s00... Dimensions>
        struct Tensor;

        template<typename T>
        struct Tensor<T> { using Type = T; };

        template<typename T, s00 Head, s00... Tail>
        struct Tensor<T, Head, Tail...>
        {

            using Type = std::array<typename Tensor<T, Tail...>::Type, Head>;

        };

    }

    template<QuantizedInteger T, s00... Dimensions>
    using NArray = Detail::Tensor<T, Dimensions...>::Type;

}

#if defined(_MSC_VER)
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#endif
