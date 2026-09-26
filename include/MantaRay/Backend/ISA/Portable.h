//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_PORTABLE_H
#define MANTARAY_BACKEND_ISA_PORTABLE_H

#include <array>
#include <bit>
#include <type_traits>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct Portable
    {

        static_assert(std::is_integral_v<T>, "ISA requires integral type");

        static_assert(std::is_signed_v<T>, "ISA requires signed integral type");

        static_assert(
            sizeof(T) == 1 ||
            sizeof(T) == 2 ||
            sizeof(T) == 4  ,
            "ISA requires integral type of size 1, 2, or 4"
        );

        using Vector = T;
        constexpr static s00 Bytes = sizeof(T);

    };

    template<typename T>
    constexpr T Add(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;
        return std::bit_cast<T>(static_cast<U>(static_cast<U>(left) + static_cast<U>(right)));
    }

    template<typename T>
    constexpr T Sub(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;
        return std::bit_cast<T>(static_cast<U>(static_cast<U>(left) - static_cast<U>(right)));
    }

    template<typename T>
    constexpr T Mul(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;
        return std::bit_cast<T>(static_cast<U>(static_cast<u64>(static_cast<U>(left)) * static_cast<U>(right)));
    }

}

#endif
