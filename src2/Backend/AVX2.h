//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_AVX2_H
#define MANTARAY_AVX2_H

#ifdef  __AVX2__

#include "AVX.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    class AVX2 : public AVX<T>
    {

        public:
        static inline Vec256I Min(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            if (std::is_same_v<T, i08>) return _mm256_min_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm256_min_epi16(ymm0, ymm1);
            if (std::is_same_v<T, i32>) return _mm256_min_epi32(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec256I Max(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            if (std::is_same_v<T, i08>) return _mm256_max_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm256_max_epi16(ymm0, ymm1);
            if (std::is_same_v<T, i32>) return _mm256_max_epi32(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec256I Add(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            if (std::is_same_v<T, i08>) return _mm256_add_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm256_add_epi16(ymm0, ymm1);
            if (std::is_same_v<T, i32>) return _mm256_add_epi32(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec256I Sub(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            if (std::is_same_v<T, i08>) return _mm256_sub_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm256_sub_epi16(ymm0, ymm1);
            if (std::is_same_v<T, i32>) return _mm256_sub_epi32(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec256I Madd(const Vec256I& ymm0, const Vec256I& ymm1) requires std::is_same_v<T, i16>
        {
            return _mm256_madd_epi16 (ymm0, ymm1);
        }

        static inline T Sum(const Vec256I& ymm0) requires std::is_same_v<T, i32>
        {
            // ymm0 = [a, b, c, d, e, f, g, h]

            Vec128I xmm0;
            Vec128I xmm1;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extracti128_si256(ymm0, 0);

            // xmm1 = [e, f, g, h]
            xmm1 = _mm256_extracti128_si256(ymm0, 1);

            //      [  a  ,   b  ,   c  ,   d  ]
            // +    [  e  ,   f  ,   g  ,   h  ]
            // =    [a + e, b + f, c + g, d + h]
            xmm0 = Add(xmm0, xmm1);

            // Refer to the SSE4::Sum method for the rest of the implementation
            // Takes a register in the form of [a, b, c, d]
            // Returns the T value of a + b + c + d
            return Sum(xmm0);
        }

    };

} // MantaRay

#endif

#endif //MANTARAY_AVX2_H
