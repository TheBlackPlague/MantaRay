//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AVX512_H
#define MANTARAY_BACKEND_ISA_AVX512_H

#if defined(__AVX512BW__)

#include <immintrin.h>

#include "AVX2.h"
#include "../../Common/Integral.h"

namespace MantaRay::Backend::ISA
{

    struct AVX512
    {

        using Vector  = __m512i;
        using VectorF = __m512 ;

        constexpr static s00 Bytes     = 64;
        constexpr static s00 Registers = 32;
        constexpr static s00 DotChains =  2;

        constexpr static bool VNNI = false;

        [[clang::always_inline]]
        static Vector Load(const void* address) { return _mm512_load_si512(address); }

        [[clang::always_inline]]
        static void Store(void* address, const Vector value) { _mm512_store_si512(address, value); }

        [[clang::always_inline]]
        static Vector Zero() { return _mm512_setzero_si512(); }

        [[clang::always_inline]]
        static Vector BroadcastI16(const i16 value) { return _mm512_set1_epi16(value); }

        [[clang::always_inline]]
        static Vector BroadcastI32(const i32 value) { return _mm512_set1_epi32(value); }

        [[clang::always_inline]]
        static Vector AddI16(const Vector left, const Vector right) { return _mm512_add_epi16(left, right); }

        [[clang::always_inline]]
        static Vector SubI16(const Vector left, const Vector right) { return _mm512_sub_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MinI16(const Vector left, const Vector right) { return _mm512_min_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MaxI16(const Vector left, const Vector right) { return _mm512_max_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI16(const Vector left, const Vector right) { return _mm512_mullo_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulHiI16(const Vector left, const Vector right) { return _mm512_mulhi_epi16(left, right); }

        [[clang::always_inline]]
        static Vector AddI32(const Vector left, const Vector right) { return _mm512_add_epi32(left, right); }

        [[clang::always_inline]]
        static Vector SubI32(const Vector left, const Vector right) { return _mm512_sub_epi32(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right) { return _mm512_mullo_epi32(left, right); }

        [[clang::always_inline]]
        static Vector MaddI16(const Vector left, const Vector right) { return _mm512_madd_epi16(left, right); }

        [[clang::always_inline]]
        static i32 ReduceI32(const Vector value)
        {
            const auto high = _mm512_extracti64x4_epi64(value, 1);

            return AVX2::ReduceI32(_mm256_add_epi32(_mm512_castsi512_si256(value), high));
        }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value)
        {
            const auto wide = _mm512_add_epi64(
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256   (value   )),
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(value, 1))
            );

            return _mm512_reduce_add_epi64(wide);
        }

        [[clang::always_inline]]
        static Vector PackUnsignedI16(const Vector left, const Vector right)
        { return _mm512_packus_epi16(left, right); }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            return _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_maddubs_epi16(u8, i8), _mm512_set1_epi16(1)));
        }

        [[clang::always_inline]]
        static VectorF LoadF(const f32* address) { return _mm512_load_ps(address); }

        [[clang::always_inline]]
        static VectorF ZeroF() { return _mm512_setzero_ps(); }

        [[clang::always_inline]]
        static VectorF AddF(const VectorF left, const VectorF right) { return _mm512_add_ps(left, right); }

        [[clang::always_inline]]
        static VectorF MulF(const VectorF left, const VectorF right) { return _mm512_mul_ps(left, right); }

        [[clang::always_inline]]
        static f32 ReduceF(const VectorF value)
        {
            const auto high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(value), 1));

            return AVX2::ReduceF(_mm256_add_ps(_mm512_castps512_ps256(value), high));
        }

    };

}

#endif

#endif //MANTARAY_BACKEND_ISA_AVX512_H
