//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __AVX__

#ifndef MANTARAY_AVX_H
#define MANTARAY_AVX_H

#include "SSE41.h"

namespace MantaRay
{

    // 256-bit integer register
    using Vec256I = __m256i;

#ifdef ALIGN
#undef ALIGN
#endif

#define ALIGN alignas(sizeof(MantaRay::Vec256I))

    template<QuantizedInteger T>
    struct AVX : SSE41<T>
    {

        constexpr static Vec256I Zero = _mm256_setzero_si256();

        static inline Vec256I From(const T value)
        {
            if (std::is_same_v<T, i08>) return _mm256_set1_epi8 (value);
            if (std::is_same_v<T, i16>) return _mm256_set1_epi16(value);
            if (std::is_same_v<T, i32>) return _mm256_set1_epi32(value);

            __builtin_unreachable();
        }

        template<s00 Size>
        static inline Vec256I From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec256I), "Array size must be at least the width of a Vec256I.");

            return _mm256_load_si256(reinterpret_cast<Vec256I const*>(&array[index]));
        }

        template<s00 Size>
        static inline void Store(const Vec256I& ymm0, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec256I), "Array size must be at least the width of a Vec256I.");

            _mm256_store_si256(reinterpret_cast<Vec256I*>(&array[index]), ymm0);
        }

        static inline Vec256I Min(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            // ymm0 = [a, b, c, d, e, f, g, h] -- Assuming T is i32
            // ymm1 = [i, j, k, l, m, n, o, p] -- Assuming T is i32

            Vec128I xmm0;
            Vec128I xmm1;
            Vec128I xmm2;
            Vec128I xmm3;

            Vec256I ymm2;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [i, j, k, l]
            xmm1 = _mm256_extractf128_si256(ymm1, 0);

            // xmm2 = [e, f, g, h]
            xmm2 = _mm256_extractf128_si256(ymm0, 1);

            // xmm3 = [m, n, o, p]
            xmm3 = _mm256_extractf128_si256(ymm1, 1);

            // xmm0 = [min(a, i), min(b, j), min(c, k), min(d, l)]
            xmm0 = SSE41<T>::Min(xmm0, xmm1);

            // xmm1 = [min(e, m), min(f, n), min(g, o), min(h, p)]
            xmm1 = SSE41<T>::Min(xmm2, xmm3);

            // ymm2 = [min(a, i), min(b, j), min(c, k), min(d, l), 0, 0, 0, 0]
            ymm2 = _mm256_insertf128_si256(Zero, xmm0, 0);

            // ymm2 = [min(a, i), min(b, j), min(c, k), min(d, l), min(e, m), min(f, n), min(g, o), min(h, p)]
            ymm2 = _mm256_insertf128_si256(ymm2, xmm1, 1);

            return ymm2;
        }

        static inline Vec256I Max(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            // ymm0 = [a, b, c, d, e, f, g, h] -- Assuming T is i32
            // ymm1 = [i, j, k, l, m, n, o, p] -- Assuming T is i32

            Vec128I xmm0;
            Vec128I xmm1;
            Vec128I xmm2;
            Vec128I xmm3;

            Vec256I ymm2;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [i, j, k, l]
            xmm1 = _mm256_extractf128_si256(ymm1, 0);

            // xmm2 = [e, f, g, h]
            xmm2 = _mm256_extractf128_si256(ymm0, 1);

            // xmm3 = [m, n, o, p]
            xmm3 = _mm256_extractf128_si256(ymm1, 1);

            // xmm0 = [max(a, i), max(b, j), max(c, k), max(d, l)]
            xmm0 = SSE41<T>::Max(xmm0, xmm1);

            // xmm1 = [max(e, m), max(f, n), max(g, o), max(h, p)]
            xmm1 = SSE41<T>::Max(xmm2, xmm3);

            // ymm2 = [max(a, i), max(b, j), max(c, k), max(d, l), 0, 0, 0, 0]
            ymm2 = _mm256_insertf128_si256(Zero, xmm0, 0);

            // ymm2 = [max(a, i), max(b, j), max(c, k), max(d, l), max(e, m), max(f, n), max(g, o), max(h, p)]
            ymm2 = _mm256_insertf128_si256(ymm2, xmm1, 1);

            return ymm2;
        }

        static inline Vec256I Add(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            // ymm0 = [a, b, c, d, e, f, g, h] -- Assuming T is i32
            // ymm1 = [i, j, k, l, m, n, o, p] -- Assuming T is i32

            Vec128I xmm0;
            Vec128I xmm1;
            Vec128I xmm2;
            Vec128I xmm3;

            Vec256I ymm2;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [i, j, k, l]
            xmm1 = _mm256_extractf128_si256(ymm1, 0);

            // xmm2 = [e, f, g, h]
            xmm2 = _mm256_extractf128_si256(ymm0, 1);

            // xmm3 = [m, n, o, p]
            xmm3 = _mm256_extractf128_si256(ymm1, 1);

            //     [  a  ,   b  ,   c  ,   d  ]
            // +   [  i  ,   j  ,   k  ,   l  ]
            // =   [a + i, b + j, c + k, d + l]
            xmm0 = SSE41<T>::Add(xmm0, xmm1);

            //     [  e  ,   f  ,   g  ,   h  ]
            // +   [  m  ,   n  ,   o  ,   p  ]
            // =   [e + m, f + n, g + o, h + p]
            xmm1 = SSE41<T>::Add(xmm2, xmm3);

            // ymm2 = [a + i, b + j, c + k, d + l, 0, 0, 0, 0]
            ymm2 = _mm256_insertf128_si256(Zero, xmm0, 0);

            // ymm2 = [a + i, b + j, c + k, d + l, e + m, f + n, g + o, h + p]
            ymm2 = _mm256_insertf128_si256(ymm2, xmm1, 1);

            return ymm2;
        }

        static inline Vec256I Sub(const Vec256I& ymm0, const Vec256I& ymm1)
        {
            // ymm0 = [a, b, c, d, e, f, g, h] -- Assuming T is i32
            // ymm1 = [i, j, k, l, m, n, o, p] -- Assuming T is i32

            Vec128I xmm0;
            Vec128I xmm1;
            Vec128I xmm2;
            Vec128I xmm3;

            Vec256I ymm2;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [i, j, k, l]
            xmm1 = _mm256_extractf128_si256(ymm1, 0);

            // xmm2 = [e, f, g, h]
            xmm2 = _mm256_extractf128_si256(ymm0, 1);

            // xmm3 = [m, n, o, p]
            xmm3 = _mm256_extractf128_si256(ymm1, 1);

            //     [  a  ,   b  ,   c  ,   d  ]
            // -   [  i  ,   j  ,   k  ,   l  ]
            // =   [a - i, b - j, c - k, d - l]
            xmm0 = SSE41<T>::Sub(xmm0, xmm1);

            //     [  e  ,   f  ,   g  ,   h  ]
            // -   [  m  ,   n  ,   o  ,   p  ]
            // =   [e - m, f - n, g - o, h - p]
            xmm1 = SSE41<T>::Sub(xmm2, xmm3);

            // ymm2 = [a - i, b - j, c - k, d - l, 0, 0, 0, 0]
            ymm2 = _mm256_insertf128_si256(Zero, xmm0, 0);

            // ymm2 = [a - i, b - j, c - k, d - l, e - m, f - n, g - o, h - p]
            ymm2 = _mm256_insertf128_si256(ymm2, xmm1, 1);

            return ymm2;
        }

        static inline Vec256I Madd(const Vec256I& ymm0, const Vec256I& ymm1) requires std::is_same_v<T, i16>
        {
            // ymm0 = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
            // ymm1 = [q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F]

            Vec128I xmm0;
            Vec128I xmm1;
            Vec128I xmm2;
            Vec128I xmm3;

            Vec256I ymm2;

            // xmm0 = [a, b, c, d, e, f, g, h]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [q, r, s, t, u, v, w, x]
            xmm1 = _mm256_extractf128_si256(ymm1, 0);

            // xmm2 = [i, j, k, l, m, n, o, p]
            xmm2 = _mm256_extractf128_si256(ymm0, 1);

            // xmm3 = [y, z, A, B, C, D, E, F]
            xmm3 = _mm256_extractf128_si256(ymm1, 1);

            //     [  a   ,   b  ,   c   ,   d  ,   e   ,   f  ,   g   ,   h  ]
            // *   [  q   ,   r  ,   s   ,   t  ,   u   ,   v  ,   w   ,   x  ]
            // =   [a * q , b * r, c * s , d * t, e * u , f * v, g * w , h * x] -- INTERMEDIATE
            // =   [a * q + b * r, c * s + d * t, e * u + f * v, g * w + h * x] -- FINAL
            xmm0 = SSE41<T>::Madd(xmm0, xmm1);

            //     [  i   ,   j  ,   k   ,   l  ,   m   ,   n  ,   o   ,   p  ]
            // *   [  y   ,   z  ,   A   ,   B  ,   C   ,   D  ,   E   ,   F  ]
            // =   [i * y , j * z, k * A , l * B, m * C , n * D, o * E , p * F] -- INTERMEDIATE
            // =   [i * y + j * z, k * A + l * B, m * C + n * D, o * E + p * F] -- FINAL
            xmm1 = SSE41<T>::Madd(xmm2, xmm3);

            // ymm2 = [a * q + b * r, c * s + d * t, e * u + f * v, g * w + h * x, 0, 0, 0, 0]
            ymm2 = _mm256_insertf128_si256(Zero, xmm0, 0);

            // ymm2 = [a * q + b * r, c * s + d * t, e * u + f * v, g * w + h * x,
            //         i * y + j * z, k * A + l * B, m * C + n * D, o * E + p * F]
            ymm2 = _mm256_insertf128_si256(ymm2, xmm1, 1);

            return ymm2;
        }

        static inline T Sum(const Vec256I& ymm0) requires std::is_same_v<T, i32>
        {
            // ymm0 = [a, b, c, d, e, f, g, h]

            Vec128I xmm0;
            Vec128I xmm1;

            T eax;
            T ebx;

            // xmm0 = [a, b, c, d]
            xmm0 = _mm256_extractf128_si256(ymm0, 0);

            // xmm1 = [e, f, g, h]
            xmm1 = _mm256_extractf128_si256(ymm0, 1);

            // Refer to the SSE4::Sum method for this part of the implementation
            // Takes a register in the form of [a, b, c, d]
            // Returns the T value of a + b + c + d
            eax = SSE41<T>::Sum(xmm0);

            // Refer to the SSE4::Sum method for this part of the implementation
            // Takes a register in the form of [a, b, c, d]
            // Returns the T value of a + b + c + d
            ebx = SSE41<T>::Sum(xmm1);

            // eax + ebx = (a + b + c + d) + (e + f + g + h) = a + b + c + d + e + f + g + h
            return eax + ebx;
        }

    };

} // MantaRay

#endif //MANTARAY_AVX_H

#endif
