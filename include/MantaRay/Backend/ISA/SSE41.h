//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_SSE41_H
#define MANTARAY_BACKEND_ISA_SSE41_H

#include "SSE2.h"

#if defined(__SSE4_1__)

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct SSE41 : SSE2<T>
    {

        using Vector = __m128i;

        [[clang::always_inline]]
        static Vector Min(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm_min_epi8 (left, right);
            if (sizeof(T) == 2) return _mm_min_epi16(left, right);
            if (sizeof(T) == 4) return _mm_min_epi32(left, right);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vector Max(const Vector left, const Vector right)
        {
            if (sizeof(T) == 1) return _mm_max_epi8 (left, right);
            if (sizeof(T) == 2) return _mm_max_epi16(left, right);
            if (sizeof(T) == 4) return _mm_max_epi32(left, right);

            __builtin_unreachable();
        }

    };

}

#endif

#endif
