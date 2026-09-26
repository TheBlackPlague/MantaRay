//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AVX_H
#define MANTARAY_BACKEND_ISA_AVX_H

#include "SSE41.h"

#if defined(__AVX__)

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct AVX : SSE41<T>
    {

        using Vector = __m256i;
        using Half   = __m128i;

        constexpr static s00 Bytes = 32;

        [[clang::always_inline]]
        static Half Low(const Vector value) { return _mm256_castsi256_si128(value); }

        [[clang::always_inline]]
        static Half High(const Vector value) { return _mm256_extractf128_si256(value, 1); }

        [[clang::always_inline]]
        static Vector Combine(const Half low, const Half high)
        { return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1); }

        [[clang::always_inline]]
        static Vector Load(const T* values)
        { return _mm256_loadu_si256(reinterpret_cast<const Vector*>(values)); }

        [[clang::always_inline]]
        static void Store(T* values, const Vector value)
        { _mm256_storeu_si256(reinterpret_cast<Vector*>(values), value); }

        [[clang::always_inline]]
        static Vector Broadcast(const T value)
        {
            const auto half = SSE41<T>::Broadcast(value);
            return Combine(half, half);
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::Add(Low (left), Low (right)),
                SSE41<T>::Add(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::Sub(Low (left), Low (right)),
                SSE41<T>::Sub(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::Min(Low (left), Low (right)),
                SSE41<T>::Min(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::Max(Low (left), Low (right)),
                SSE41<T>::Max(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector MulLo(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::MulLo(Low (left), Low (right)),
                SSE41<T>::MulLo(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector MulHighUnsigned(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::MulHighUnsigned(Low (left), Low (right)),
                SSE41<T>::MulHighUnsigned(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector MultiplyAddPairs(const Vector left, const Vector right)
        {
            return Combine(
                SSE41<T>::MultiplyAddPairs(Low (left), Low (right)),
                SSE41<T>::MultiplyAddPairs(High(left), High(right))
            );
        }

        template<s00 Bits>
        [[clang::always_inline]]
        static Vector ShiftRight(const Vector value)
        {
            return Combine(
                SSE41<T>::template ShiftRight<Bits>(Low (value)),
                SSE41<T>::template ShiftRight<Bits>(High(value))
            );
        }

        [[clang::always_inline]]
        static i32 Sum(const Vector value)
        {
            return SSE41<i32>::Sum(_mm_add_epi32(Low(value), High(value)));
        }

    };

}

#endif

#endif
