//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_SSE4_H
#define MANTARAY_SSE4_H

#include "SSE2.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    class SSE4 : public SSE2<T>
    {

        public:
        static inline T Sum(const Vec128I& xmm0) requires std::is_same_v<T, i32>
        {
            // xmm0 = [a, b, c, d]

            Vec128I xmm1;

            T eax;
            T ebx;

            // xmm1 = [c, d, c, d]
            xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);

            //     [  a  ,   b  ,   c  ,   d  ]
            // +   [  c  ,   d  ,   c  ,   d  ]
            // =   [a + c, b + d, c + c, d + d]
            xmm1 = _mm_add_epi32(xmm0, xmm1);

            // eax = a + c
            eax = _mm_extract_epi32(xmm1, 0);

            // ebx = b + d
            ebx = _mm_extract_epi32(xmm1, 1);

            // (a + c) + (b + d) = a + c + b + d = a + b + c + d
            return eax + ebx;
        }

    };

} // MantaRay

#endif //MANTARAY_SSE4_H
