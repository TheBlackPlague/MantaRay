//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_NEON_H
#define MANTARAY_BACKEND_ISA_NEON_H

#include "ARM64.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct NEON : ARM64<T>
    {

        using V8_16 = int8x16_t;
        using V16_8 = int16x8_t;
        using V32_4 = int32x4_t;

        using Vector = V32_4;

        constexpr static s00 Bytes = 16;

        [[clang::always_inline]]
        static Vector Load(const T* values)
        { return std::bit_cast<Vector>(vld1q_u8(reinterpret_cast<const u08*>(values))); }

        [[clang::always_inline]]
        static void Store(T* values, const Vector value)
        { vst1q_u8(reinterpret_cast<u08*>(values), std::bit_cast<uint8x16_t>(value)); }

        [[clang::always_inline]]
        static Vector Broadcast(const T value)
        {
            if (sizeof(T) == 1) return std::bit_cast<Vector>(vdupq_n_s8 (value));
            if (sizeof(T) == 2) return std::bit_cast<Vector>(vdupq_n_s16(value));
            if (sizeof(T) == 4) return vdupq_n_s32(value);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1)
                return std::bit_cast<Vector>(vaddq_s8 (std::bit_cast<V8_16>(left), std::bit_cast<V8_16>(right)));
            if (sizeof(T) == 2)
                return std::bit_cast<Vector>(vaddq_s16(std::bit_cast<V16_8>(left), std::bit_cast<V16_8>(right)));
            if (sizeof(T) == 4) return vaddq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1)
                return std::bit_cast<Vector>(vsubq_s8 (std::bit_cast<V8_16>(left), std::bit_cast<V8_16>(right)));
            if (sizeof(T) == 2)
                return std::bit_cast<Vector>(vsubq_s16(std::bit_cast<V16_8>(left), std::bit_cast<V16_8>(right)));
            if (sizeof(T) == 4) return vsubq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1)
                return std::bit_cast<Vector>(vminq_s8 (std::bit_cast<V8_16>(left), std::bit_cast<V8_16>(right)));
            if (sizeof(T) == 2)
                return std::bit_cast<Vector>(vminq_s16(std::bit_cast<V16_8>(left), std::bit_cast<V16_8>(right)));
            if (sizeof(T) == 4) return vminq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1)
                return std::bit_cast<Vector>(vmaxq_s8 (std::bit_cast<V8_16>(left), std::bit_cast<V8_16>(right)));
            if (sizeof(T) == 2)
                return std::bit_cast<Vector>(vmaxq_s16(std::bit_cast<V16_8>(left), std::bit_cast<V16_8>(right)));
            if (sizeof(T) == 4) return vmaxq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulLo(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1)
                return std::bit_cast<Vector>(vmulq_s8 (std::bit_cast<V8_16>(left), std::bit_cast<V8_16>(right)));
            if (sizeof(T) == 2)
                return std::bit_cast<Vector>(vmulq_s16(std::bit_cast<V16_8>(left), std::bit_cast<V16_8>(right)));
            if (sizeof(T) == 4) return vmulq_s32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulHighUnsigned(const Vector left, const Vector right)
        {
            const auto a = std::bit_cast<uint16x8_t>(left );
            const auto b = std::bit_cast<uint16x8_t>(right);

            return std::bit_cast<Vector>(vcombine_u16(
                vshrn_n_u32(vmull_u16(vget_low_u16(a), vget_low_u16(b)), 16),
                vshrn_n_u32(vmull_high_u16(a, b), 16)
            ));
        }

        template<s00 Bits>
        [[clang::always_inline]]
        static Vector ShiftRight(const Vector value)
        {
            using VU8_16 = uint8x16_t;
            using VU16_8 = uint16x8_t;
            using VU32_4 = uint32x4_t;

            if constexpr (Bits == 0) return value;
            else {
                if (sizeof(T) == 1) return std::bit_cast<Vector>(
                    vshlq_u8 (std::bit_cast<VU8_16>(value), vdupq_n_s8 (-static_cast<i08>(Bits)))
                );
                if (sizeof(T) == 2) return std::bit_cast<Vector>(
                    vshlq_u16(std::bit_cast<VU16_8>(value), vdupq_n_s16(-static_cast<i16>(Bits)))
                );
                if (sizeof(T) == 4) return std::bit_cast<Vector>(
                    vshlq_u32(std::bit_cast<VU32_4>(value), vdupq_n_s32(-static_cast<i32>(Bits)))
                );
            }

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MultiplyAddPairs(const Vector left, const Vector right)
        {
            const auto a = std::bit_cast<V16_8>(left );
            const auto b = std::bit_cast<V16_8>(right);

            return vpaddq_s32(vmull_s16(vget_low_s16(a), vget_low_s16(b)), vmull_high_s16(a, b));
        }

        [[clang::always_inline]]
        static i32 Sum(const Vector value) { return vaddvq_s32(value); }

    };

}

#endif

#endif
