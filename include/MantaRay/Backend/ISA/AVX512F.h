//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AVX512F_H
#define MANTARAY_BACKEND_ISA_AVX512F_H

#include "AVX2.h"

#if defined(__AVX512F__)

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct AVX512F : AVX2<T>
    {

        constexpr static s00 Bytes = 64;

        using Vector = __m512i;
        using Half   = __m256i;

        [[clang::always_inline]]
        static Half Low(const Vector value) { return _mm512_castsi512_si256(value); }

        [[clang::always_inline]]
        static Half High(const Vector value) { return _mm512_extracti64x4_epi64(value, 1); }

        [[clang::always_inline]]
        static Vector Combine(const Half low, const Half high)
        { return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1); }

        [[clang::always_inline]]
        static Vector Load(const T* values) { return _mm512_loadu_si512(reinterpret_cast<const Vector*>(values)); }

        [[clang::always_inline]]
        static void Store(T* values, const Vector value)
        { _mm512_storeu_si512(reinterpret_cast<Vector*>(values), value); }

        [[clang::always_inline]]
        static Vector Broadcast(const T value)
        {
            const auto half = AVX2<T>::Broadcast(value);
            return Combine(half, half);
        }

        [[clang::always_inline]]
        static Vector Add(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::Add(Low (left), Low (right)),
                    AVX2<T>::Add(High(left), High(right))
                );
            }
            if (sizeof(T) == 4) return _mm512_add_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Sub(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::Sub(Low (left), Low (right)),
                    AVX2<T>::Sub(High(left), High(right))
                );
            }
            if (sizeof(T) == 4) return _mm512_sub_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::Min(Low (left), Low (right)),
                    AVX2<T>::Min(High(left), High(right))
                );
            }
            if (sizeof(T) == 4) return _mm512_min_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::Max(Low (left), Low (right)),
                    AVX2<T>::Max(High(left), High(right))
                );
            }
            if (sizeof(T) == 4) return _mm512_max_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulLo(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::MulLo(Low (left), Low (right)),
                    AVX2<T>::MulLo(High(left), High(right))
                );
            }
            if (sizeof(T) == 4) return _mm512_mullo_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector MulHighUnsigned(const Vector left, const Vector right)
        {
            return Combine(
                AVX2<T>::MulHighUnsigned(Low (left), Low (right)),
                AVX2<T>::MulHighUnsigned(High(left), High(right))
            );
        }

        [[clang::always_inline]]
        static Vector MultiplyAddPairs(const Vector left, const Vector right)
        {
            return Combine(
                AVX2<T>::MultiplyAddPairs(Low (left), Low (right)),
                AVX2<T>::MultiplyAddPairs(High(left), High(right))
            );
        }

        template<s00 Bits>
        [[clang::always_inline]]
        static Vector ShiftRight(const Vector value)
        {
            if (sizeof(T) == 1 ||
                sizeof(T) == 2  ) {
                return Combine(
                    AVX2<T>::template ShiftRight<Bits>(Low (value)),
                    AVX2<T>::template ShiftRight<Bits>(High(value))
                );
            }
            if (sizeof(T) == 4) return _mm512_srli_epi32(value, Bits);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static i32 Sum(const Vector value)
        {
            return _mm512_reduce_add_epi32(value);
        }

    };

}

#endif

#endif
