//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __SSE4_1__

#ifndef MANTARAY_SSE4_H
#define MANTARAY_SSE4_H

#include "SSE2.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    struct SSE41 : SSE2<T>
    {

        static inline Vec128I Min(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i08>) return _mm_min_epi8 (xmm0, xmm1);
            if (std::is_same_v<T, i32>) return _mm_min_epi32(xmm0, xmm1);

            if (std::is_same_v<T, i16>) return SSE2<T>::Min(xmm0, xmm1);

            __builtin_unreachable();
        }

        static inline Vec128I Max(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i08>) return _mm_max_epi8 (xmm0, xmm1);
            if (std::is_same_v<T, i32>) return _mm_max_epi32(xmm0, xmm1);

            if (std::is_same_v<T, i16>) return SSE2<T>::Max(xmm0, xmm1);

            __builtin_unreachable();
        }

        static inline T Sum(const Vec128I& xmm0) requires std::is_same_v<T, i32>
        {
            // xmm0 = [a, b, c, d] -- Assuming T is i32

            Vec128I xmm1;

            T eax;
            T ebx;

            // xmm1 = [c, d, c, d]
            xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);

            //     [  a  ,   b  ,   c  ,   d  ]
            // +   [  c  ,   d  ,   c  ,   d  ]
            // =   [a + c, b + d, c + c, d + d]
            xmm1 = SSE2<T>::Add(xmm0, xmm1);

            // eax = a + c
            eax = _mm_extract_epi32(xmm1, 0);

            // ebx = b + d
            ebx = _mm_extract_epi32(xmm1, 1);

            // eax + ebx = (a + c) + (b + d) = a + c + b + d = a + b + c + d
            return eax + ebx;
        }

    };

} // MantaRay

#endif //MANTARAY_SSE4_H

#endif
