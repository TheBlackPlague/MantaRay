//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_SSE2_H
#define MANTARAY_BACKEND_ISA_SSE2_H

#if defined(__SSE2__)

#include <immintrin.h>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::ISA
{

    struct SSE2
    {

        using Vector  = __m128i;
        using VectorF = __m128 ;

        constexpr static s00 Bytes     = 16;
        constexpr static s00 Registers = 16;
        constexpr static s00 DotChains =  2;

        constexpr static bool VNNI = false;

        [[clang::always_inline]]
        static Vector Load(const void* address) { return _mm_load_si128(static_cast<const Vector*>(address)); }

        [[clang::always_inline]]
        static void Store(void* address, const Vector value) { _mm_store_si128(static_cast<Vector*>(address), value); }

        [[clang::always_inline]]
        static Vector Zero() { return _mm_setzero_si128(); }

        [[clang::always_inline]]
        static Vector BroadcastI16(const i16 value) { return _mm_set1_epi16(value); }

        [[clang::always_inline]]
        static Vector BroadcastI32(const i32 value) { return _mm_set1_epi32(value); }

        [[clang::always_inline]]
        static Vector AddI16(const Vector left, const Vector right) { return _mm_add_epi16(left, right); }

        [[clang::always_inline]]
        static Vector SubI16(const Vector left, const Vector right) { return _mm_sub_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MinI16(const Vector left, const Vector right) { return _mm_min_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MaxI16(const Vector left, const Vector right) { return _mm_max_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI16(const Vector left, const Vector right) { return _mm_mullo_epi16(left, right); }

        [[clang::always_inline]]
        static Vector MulHiI16(const Vector left, const Vector right) { return _mm_mulhi_epi16(left, right); }

        [[clang::always_inline]]
        static Vector AddI32(const Vector left, const Vector right) { return _mm_add_epi32(left, right); }

        [[clang::always_inline]]
        static Vector SubI32(const Vector left, const Vector right) { return _mm_sub_epi32(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right)
        {
            const auto even = _mm_mul_epu32(               left    ,                right    );
            const auto odd  = _mm_mul_epu32(_mm_srli_si128(left, 4), _mm_srli_si128(right, 4));

            return _mm_unpacklo_epi32(_mm_shuffle_epi32(even, 0x88), _mm_shuffle_epi32(odd, 0x88));
        }

        [[clang::always_inline]]
        static Vector MaddI16(const Vector left, const Vector right) { return _mm_madd_epi16(left, right); }

        [[clang::always_inline]]
        static i32 ReduceI32(const Vector value)
        {
            const auto pairs = _mm_add_epi32(value, _mm_unpackhi_epi64(value, value));

            return _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_shuffle_epi32(pairs, 0x55)));
        }

        [[clang::always_inline]]
        static i64 ReducePairI64(const Vector value)
        { return _mm_cvtsi128_si64(_mm_add_epi64(value, _mm_unpackhi_epi64(value, value))); }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value)
        {
            const auto sign = _mm_srai_epi32(value, 31);
            const auto wide = _mm_add_epi64(_mm_unpacklo_epi32(value, sign), _mm_unpackhi_epi32(value, sign));

            return ReducePairI64(wide);
        }

        [[clang::always_inline]]
        static Vector PackUnsignedI16(const Vector left, const Vector right) { return _mm_packus_epi16(left, right); }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            const auto zero = _mm_setzero_si128();

            const auto low  = _mm_srai_epi16(_mm_unpacklo_epi8(zero, i8), 8);
            const auto high = _mm_srai_epi16(_mm_unpackhi_epi8(zero, i8), 8);

            const auto pairs = _mm_packs_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi8(u8, zero), low ),
                _mm_madd_epi16(_mm_unpackhi_epi8(u8, zero), high)
            );

            return _mm_add_epi32(sum, _mm_madd_epi16(pairs, _mm_set1_epi16(1)));
        }

        [[clang::always_inline]]
        static VectorF LoadF(const f32* address) { return _mm_load_ps(address); }

        [[clang::always_inline]]
        static VectorF ZeroF() { return _mm_setzero_ps(); }

        [[clang::always_inline]]
        static VectorF AddF(const VectorF left, const VectorF right) { return _mm_add_ps(left, right); }

        [[clang::always_inline]]
        static VectorF MulF(const VectorF left, const VectorF right) { return _mm_mul_ps(left, right); }

        [[clang::always_inline]]
        static f32 ReduceF(const VectorF value)
        {
            const auto pairs = _mm_add_ps(value, _mm_movehl_ps(value, value));

            return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55)));
        }

    };

}

#endif

#endif //MANTARAY_BACKEND_ISA_SSE2_H
