//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AVX512_H
#define MANTARAY_BACKEND_ISA_AVX512_H

#include "AVX512F.h"

#if defined(__AVX512BW__)

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct AVX512 : AVX512F<T>
    {

        using Vector = __m512i;

        constexpr static s00 Bytes = 64;

        [[clang::always_inline]]
        static Vector Load(const T* values) { return _mm512_loadu_si512(reinterpret_cast<const Vector*>(values)); }

        [[clang::always_inline]]
        static void Store(T* values, const Vector value)
        { _mm512_storeu_si512(reinterpret_cast<Vector*>(values), value); }

        [[clang::always_inline]]
        static Vector Broadcast(const T value)
        {
            if (sizeof(T) == 1) return _mm512_set1_epi8 (value);
            if (sizeof(T) == 2) return _mm512_set1_epi16(value);
            if (sizeof(T) == 4) return _mm512_set1_epi32(value);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm512_add_epi8 (left, right);
            if (sizeof(T) == 2) return _mm512_add_epi16(left, right);
            if (sizeof(T) == 4) return _mm512_add_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm512_sub_epi8 (left, right);
            if (sizeof(T) == 2) return _mm512_sub_epi16(left, right);
            if (sizeof(T) == 4) return _mm512_sub_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm512_min_epi8 (left, right);
            if (sizeof(T) == 2) return _mm512_min_epi16(left, right);
            if (sizeof(T) == 4) return _mm512_min_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm512_max_epi8 (left, right);
            if (sizeof(T) == 2) return _mm512_max_epi16(left, right);
            if (sizeof(T) == 4) return _mm512_max_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulLo(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) {
                const auto mask = _mm512_set1_epi16(255);

                const auto low = _mm512_and_si512(_mm512_mullo_epi16(left, right), mask);

                const auto high = _mm512_slli_epi16(
                    _mm512_mullo_epi16(
                        _mm512_srli_epi16(left , 8),
                        _mm512_srli_epi16(right, 8)
                    ),
                    8
                );

                return _mm512_or_si512(low, high);
            }

            if (sizeof(T) == 2) return _mm512_mullo_epi16(left, right);
            if (sizeof(T) == 4) return _mm512_mullo_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulHighUnsigned(const Vector left, const Vector right) { return _mm512_mulhi_epu16(left, right); }

        template<s00 Bits>
        [[clang::always_inline]]
        static Vector ShiftRight(const Vector value)
        {
            if (sizeof(T) == 1) {
                return _mm512_and_si512(
                    _mm512_srli_epi16(value, Bits),
                    _mm512_set1_epi8(static_cast<char>(255U >> Bits))
                );
            }
            if (sizeof(T) == 2) return _mm512_srli_epi16(value, Bits);
            if (sizeof(T) == 4) return _mm512_srli_epi32(value, Bits);
        }

        [[clang::always_inline]]
        static Vector MultiplyAddPairs(const Vector left, const Vector right) { return _mm512_madd_epi16(left, right); }

        [[clang::always_inline]]
        static i32 Sum(const Vector value)
        {
            return _mm512_reduce_add_epi32(value);
        }

    };

}

#endif

#endif
