//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_INTEGRAL_H
#define MANTARAY_INTEGRAL_H

#include <cstddef>
#include <cstdint>

namespace MantaRay
{

    using s00 = std::size_t;

    using u64 = std::uint64_t;
    using i64 =  std::int64_t;

    using u32 = std::uint32_t;
    using i32 =  std::int32_t;

    using u16 = std::uint16_t;
    using i16 =  std::int16_t;

    using u08 = std::uint8_t;
    using i08 =  std::int8_t;

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapAdd(const T lhs, const T rhs)
    {
        T result;
        __builtin_add_overflow(lhs, rhs, &result);
        return result;
    }

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapSub(const T lhs, const T rhs)
    {
        T result;
        __builtin_sub_overflow(lhs, rhs, &result);
        return result;
    }

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapMul(const T lhs, const T rhs)
    {
        T result;
        __builtin_mul_overflow(lhs, rhs, &result);
        return result;
    }

}

#endif
