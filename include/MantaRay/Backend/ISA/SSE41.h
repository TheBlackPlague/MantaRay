//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_SSE41_H
#define MANTARAY_BACKEND_ISA_SSE41_H

#if defined(__SSE4_1__)

#include <immintrin.h>

#include "SSE2.h"
#include "../../Common.h"

namespace MantaRay::Backend::ISA
{

    struct SSE41 : SSE2
    {

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right) { return _mm_mullo_epi32(left, right); }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value)
        {
            const auto wide = _mm_add_epi64(
                _mm_cvtepi32_epi64(                   value        ),
                _mm_cvtepi32_epi64(_mm_unpackhi_epi64(value, value))
            );

            return ReducePairI64(wide);
        }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            return _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(u8, i8), _mm_set1_epi16(1)));
        }

    };

}

#endif

#endif //MANTARAY_BACKEND_ISA_SSE41_H
