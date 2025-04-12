//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_AVX512F_H
#define MANTARAY_AVX512F_H

#include "AVX2.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    class AVX512F : public AVX2<T>
    {

        public:
        constexpr static Vec512I Zero = _mm512_setzero_si512();

        static inline Vec512I From(const T value)
        {
            if (std::is_same_v<T, i08>) return _mm512_set1_epi8 (value);
            if (std::is_same_v<T, i16>) return _mm512_set1_epi16(value);
            if (std::is_same_v<T, i32>) return _mm512_set1_epi32(value);

            __builtin_unreachable();
        }

        template<s00 Size>
        static inline Vec512I From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec512I), "Array size must be at least the width of a Vec512I.");

            return _mm512_load_si512(reinterpret_cast<Vec512I const*>(&array[index]));
        }

        template<s00 Size>
        static inline void Store(const Vec512I& zmm0, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec512I), "Array size must be at least the width of a Vec512I.");

            _mm512_store_si512(reinterpret_cast<Vec512I*>(&array[index]), zmm0);
        }

        static inline Vec512I Min(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i32>
        {
            return _mm512_min_epi32(zmm0, zmm1);
        }

        static inline Vec512I Max(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i32>
        {
            return _mm512_max_epi32(zmm0, zmm1);
        }

        static inline Vec512I Add(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i32>
        {
            return _mm512_add_epi32(zmm0, zmm1);
        }

        static inline Vec512I Sub(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i32>
        {
            return _mm512_sub_epi32(zmm0, zmm1);
        }

        static inline T Sum(const Vec512I& zmm0) requires std::is_same_v<T, i32>
        {
            // zmm0 = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]

            Vec256I ymm0;
            Vec256I ymm1;

            // ymm0 = [a, b, c, d, e, f, g, h]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [i, j, k, l, m, n, o, p]
            ymm1 = _mm512_extracti64x4_epi64(zmm0, 1);

            //     [  a  ,   b  ,   c  ,   d  ,   e  ,   f  ,   g  ,   h  ]
            // +   [  i  ,   j  ,   k  ,   l  ,   m  ,   n  ,   o  ,   p  ]
            // =   [a + i, b + j, c + k, d + l, e + m, f + n, g + o, h + p]
            ymm0 = Add(ymm0, ymm1);

            // Refer to the AVX2::Sum method for the rest of the implementation
            // Takes a register in the form of [a, b, c, d, e, f, g, h]
            // Returns the T value of a + b + c + d + e + f + g + h
            return Sum(ymm0);
        }

    };

} // MantaRay

#endif //MANTARAY_AVX512F_H
