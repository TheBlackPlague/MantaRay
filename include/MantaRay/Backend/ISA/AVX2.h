//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AVX2_H
#define MANTARAY_BACKEND_ISA_AVX2_H

#if defined(__AVX2__)

#include <array>

#include <immintrin.h>

#include "SSE41.h"
#include "../../Common.h"

namespace MantaRay::Backend::ISA
{

    struct AVX2
    {

        using Vector  = __m256i;
        using VectorF = __m256 ;

        constexpr static s00 Bytes     = 32;
        constexpr static s00 Registers = 16;
        constexpr static s00 DotChains =  2;

        constexpr static bool VNNI = false;

        template<typename T, s00 N>
        [[clang::always_inline]]
        static Vector Load(const std::array<T, N>& array, const s00 index)
        { return _mm256_load_si256(reinterpret_cast<const Vector*>(&array[index])); }

        template<typename T, s00 N>
        [[clang::always_inline]]
        static void Store(std::array<T, N>& array, const s00 index, const Vector value)
        { _mm256_store_si256(reinterpret_cast<Vector*>(&array[index]), value); }

        [[clang::always_inline]]
        static Vector Zero() { return _mm256_setzero_si256(); }

        [[clang::always_inline]]
        static Vector BroadcastI16(const i16 value) { return _mm256_set1_epi16(value); }

        [[clang::always_inline]]
        static Vector BroadcastI32(const i32 value) { return _mm256_set1_epi32(value); }

        [[clang::always_inline]]
        static Vector AddI16(const Vector left, const Vector right) { return _mm256_add_epi16(left, right); }

        [[clang::always_inline]]
        static Vector SubI16(const Vector left, const Vector right) { return _mm256_sub_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MinI16(const Vector left, const Vector right) { return _mm256_min_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MaxI16(const Vector left, const Vector right) { return _mm256_max_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI16(const Vector left, const Vector right) { return _mm256_mullo_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulHiI16(const Vector left, const Vector right) { return _mm256_mulhi_epi16(left, right); }

        [[clang::always_inline]]
        static Vector AddI32(const Vector left, const Vector right) { return _mm256_add_epi32(left, right); }

        [[clang::always_inline]]
        static Vector SubI32(const Vector left, const Vector right) { return _mm256_sub_epi32(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right) { return _mm256_mullo_epi32(left, right); }

        [[clang::always_inline]]
        static Vector MaddI16(const Vector left, const Vector right) { return _mm256_madd_epi16(left, right); }

        [[clang::always_inline]]
        static i32 ReduceI32(const Vector value)
        {
            return SSE41::ReduceI32(_mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1)));
        }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value)
        {
            const auto wide = _mm256_add_epi64(
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128  (value   )),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(value, 1))
            );

            const auto half = _mm_add_epi64(_mm256_castsi256_si128(wide), _mm256_extracti128_si256(wide, 1));

            return SSE41::ReducePairI64(half);
        }

        [[clang::always_inline]]
        static Vector PackUnsignedI16(const Vector left, const Vector right)
        { return _mm256_packus_epi16(left, right); }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            return _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(u8, i8), _mm256_set1_epi16(1)));
        }

        template<s00 N>
        [[clang::always_inline]]
        static VectorF LoadF(const std::array<f32, N>& array, const s00 index) { return _mm256_load_ps(&array[index]); }

        [[clang::always_inline]]
        static VectorF ZeroF() { return _mm256_setzero_ps(); }

        [[clang::always_inline]]
        static VectorF AddF(const VectorF left, const VectorF right) { return _mm256_add_ps(left, right); }

        [[clang::always_inline]]
        static VectorF MulF(const VectorF left, const VectorF right) { return _mm256_mul_ps(left, right); }

        [[clang::always_inline]]
        static f32 ReduceF(const VectorF value)
        {
            return SSE41::ReduceF(_mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1)));
        }

    };

}

#endif

#endif //MANTARAY_BACKEND_ISA_AVX2_H
