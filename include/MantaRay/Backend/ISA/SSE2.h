//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_SSE2_H
#define MANTARAY_BACKEND_ISA_SSE2_H

#include "AMD64.h"

#if defined(__SSE2__)

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct SSE2 : AMD64<T>
    {

        using Vector = __m128i;

        constexpr static s00 Bytes = 16;

        [[clang::always_inline]]
        static Vector Load(const T* values) { return _mm_loadu_si128(reinterpret_cast<const Vector*>(values)); }

        [[clang::always_inline]]
        static void Store(T* values, const Vector value) { _mm_storeu_si128(reinterpret_cast<Vector*>(values), value); }

        [[clang::always_inline]]
        static Vector Broadcast(const T value)
        {
            if (sizeof(T) == 1) return _mm_set1_epi8 (value);
            if (sizeof(T) == 2) return _mm_set1_epi16(value);
            if (sizeof(T) == 4) return _mm_set1_epi32(value);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm_add_epi8 (left, right);
            if (sizeof(T) == 2) return _mm_add_epi16(left, right);
            if (sizeof(T) == 4) return _mm_add_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm_sub_epi8 (left, right);
            if (sizeof(T) == 2) return _mm_sub_epi16(left, right);
            if (sizeof(T) == 4) return _mm_sub_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (sizeof(T) == 2) return _mm_min_epi16(left, right);

            const auto mask = sizeof(T) == 1 ?
                _mm_cmpgt_epi8 (left, right) :
                _mm_cmpgt_epi32(left, right) ;

            return _mm_or_si128(_mm_and_si128(mask, right), _mm_andnot_si128(mask, left));
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (sizeof(T) == 2) return _mm_max_epi16(left, right);

            const auto mask = sizeof(T) == 1 ?
                _mm_cmpgt_epi8 (left, right) :
                _mm_cmpgt_epi32(left, right) ;

            return _mm_or_si128(_mm_and_si128(mask, left), _mm_andnot_si128(mask, right));
        }

        [[clang::always_inline]]
        static Vector MulLo(const Vector left, const Vector right)
        {
            if (sizeof(T) == 2) return _mm_mullo_epi16(left, right);

            if (sizeof(T) == 1) {
                const auto mask = _mm_set1_epi16(255);

                const auto low = _mm_and_si128(_mm_mullo_epi16(left, right), mask);

                const auto high = _mm_slli_epi16(
                    _mm_mullo_epi16(
                        _mm_srli_epi16( left, 8),
                        _mm_srli_epi16(right, 8)),
                    8
                );

                return _mm_or_si128(low, high);
            }

            const auto even = _mm_mul_epu32(               left,                    right    );
            const auto odd  = _mm_mul_epu32(_mm_srli_si128(left, 4), _mm_srli_si128(right, 4));

            return _mm_unpacklo_epi32(_mm_shuffle_epi32(even, 0x88), _mm_shuffle_epi32(odd, 0x88));
        }

        [[clang::always_inline]]
        static Vector MulHighUnsigned(const Vector left, const Vector right) { return _mm_mulhi_epu16(left, right); }

        template<s00 Bits>
        [[clang::always_inline]]
        static Vector ShiftRight(const Vector value)
        {
            if (sizeof(T) == 1)
                return _mm_and_si128(_mm_srli_epi16(value, Bits), _mm_set1_epi8(static_cast<char>(255U >> Bits)));

            if (sizeof(T) == 2) return _mm_srli_epi16(value, Bits);
            if (sizeof(T) == 4) return _mm_srli_epi32(value, Bits);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MultiplyAddPairs(const Vector left, const Vector right)
        {
            static_assert(sizeof(T) == 2, "Multiply Add Pairs is only implemented for 16-bit integers");

            return _mm_madd_epi16(left, right);
        }

        [[clang::always_inline]]
        static i32 Sum(const Vector value)
        {
            auto sum = _mm_add_epi32(value, _mm_unpackhi_epi64(value, value));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x55));
            return _mm_cvtsi128_si32(sum);
        }

    };

}

#endif

#endif
