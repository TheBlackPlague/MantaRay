//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_NEON_H
#define MANTARAY_NEON_H

#include <arm_neon.h>
#include <bit>

#include "../../Common/Container.h"
#include "../../Common/Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    struct NEON
    {

        using Vector = int32x4_t;

        constexpr static Vector Zero {};

        [[clang::always_inline]]
        static Vector From(const T value)
        {
            if (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vdupq_n_s8 (value));
            if (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vdupq_n_s16(value));
            if (std::is_same_v<T, i32>) return                       vdupq_n_s32(value) ;

            __builtin_unreachable();
        }

        template<s00 Size>
        [[clang::always_inline]]
        static Vector From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vector));

            if constexpr (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vld1q_s8 (array.data() + index));
            if constexpr (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vld1q_s16(array.data() + index));
            if constexpr (std::is_same_v<T, i32>) return                       vld1q_s32(array.data() + index) ;

            __builtin_unreachable();
        }

        template<s00 Size>
        [[clang::always_inline]]
        static void Store(const Vector value, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vector));

            if constexpr (std::is_same_v<T, i08>) vst1q_s8 (array.data() + index, std::bit_cast<int8x16_t>(value));
            if constexpr (std::is_same_v<T, i16>) vst1q_s16(array.data() + index, std::bit_cast<int16x8_t>(value));
            if constexpr (std::is_same_v<T, i32>) vst1q_s32(array.data() + index,                          value );
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vminq_s8 (
                std::bit_cast<int8x16_t>(left), std::bit_cast<int8x16_t>(right)));
            if (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vminq_s16(
                std::bit_cast<int16x8_t>(left), std::bit_cast<int16x8_t>(right)));
            if (std::is_same_v<T, i32>) return vminq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vmaxq_s8 (
                std::bit_cast<int8x16_t>(left), std::bit_cast<int8x16_t>(right)));
            if (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vmaxq_s16(
                std::bit_cast<int16x8_t>(left), std::bit_cast<int16x8_t>(right)));
            if (std::is_same_v<T, i32>) return vmaxq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            if (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vaddq_s8 (
                std::bit_cast<int8x16_t>(left), std::bit_cast<int8x16_t>(right)));
            if (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vaddq_s16(
                std::bit_cast<int16x8_t>(left), std::bit_cast<int16x8_t>(right)));
            if (std::is_same_v<T, i32>) return vaddq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            if (std::is_same_v<T, i08>) return std::bit_cast<Vector>(vsubq_s8 (
                std::bit_cast<int8x16_t>(left), std::bit_cast<int8x16_t>(right)));
            if (std::is_same_v<T, i16>) return std::bit_cast<Vector>(vsubq_s16(
                std::bit_cast<int16x8_t>(left), std::bit_cast<int16x8_t>(right)));
            if (std::is_same_v<T, i32>) return vsubq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static T Sum(const Vector value) requires std::is_same_v<T, i32>
        {
            return vaddvq_s32(value);
        }

    };

}

#endif
