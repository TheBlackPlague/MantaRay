//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_INTEGRAL_H
#define MANTARAY_COMMON_INTEGRAL_H

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace MantaRay
{

    using s00 = std::size_t;

    using u64 = std::uint64_t;
    using i64 = std:: int64_t;

    using u32 = std::uint32_t;
    using i32 = std:: int32_t;

    using u16 = std::uint16_t;
    using i16 = std:: int16_t;

    using u08 = std::uint8_t;
    using i08 = std:: int8_t;

    template<typename T>
    constexpr T WrapAdd(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;

        return std::bit_cast<T>(static_cast<U>(static_cast<u64>(static_cast<U>(left)) + static_cast<U>(right)));
    }

    template<typename T>
    constexpr T WrapSub(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;

        return std::bit_cast<T>(static_cast<U>(static_cast<u64>(static_cast<U>(left)) - static_cast<U>(right)));
    }

    template<typename T>
    constexpr T WrapMul(const T left, const T right)
    {
        using U = std::make_unsigned_t<T>;

        return std::bit_cast<T>(static_cast<U>(static_cast<u64>(static_cast<U>(left)) * static_cast<U>(right)));
    }

}

#endif
