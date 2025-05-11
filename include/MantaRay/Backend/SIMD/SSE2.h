//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __SSE2__

#ifndef MANTARAY_SSE2_H
#define MANTARAY_SSE2_H

#include "AMD64.h"

#include "../Container.h"

namespace MantaRay
{

    // 128-bit integer register
    using Vec128I = __m128i;

#ifdef ALIGN
#undef ALIGN
#endif

#define ALIGN alignas(sizeof(MantaRay::Vec128I))

    template<QuantizedInteger T>
    struct SSE2 : AMD64<T>
    {

        constexpr static Vec128I Zero = _mm_setzero_si128();

        [[clang::always_inline]]
        static Vec128I From(const T value)
        {
            if (std::is_same_v<T, i08>) return _mm_set1_epi8 (value);
            if (std::is_same_v<T, i16>) return _mm_set1_epi16(value);
            if (std::is_same_v<T, i32>) return _mm_set1_epi32(value);

            __builtin_unreachable();
        }

        template<s00 Size>
        [[clang::always_inline]]
        static Vec128I From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec128I), "Array size must be at least the width of a Vec128I.");

            return _mm_load_si128(reinterpret_cast<Vec128I const*>(&array[index]));
        }

        template<s00 Size>
        [[clang::always_inline]]
        static void Store(const Vec128I& xmm0, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec128I), "Array size must be at least the width of a Vec128I.");

            _mm_store_si128(reinterpret_cast<Vec128I*>(&array[index]), xmm0);
        }

        [[clang::always_inline]]
        static Vec128I Min(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i16>) return _mm_min_epi16(xmm0, xmm1);

            constexpr static s00 Size = sizeof(Vec128I) / sizeof(T);

            ALIGN Array<T, Size> xmm0Array;
            ALIGN Array<T, Size> xmm1Array;
            ALIGN Array<T, Size> xmm2Array;

            Store(xmm0, xmm0Array, 0);
            Store(xmm1, xmm1Array, 0);

            for (s00 i = 0; i < Size; i++) xmm2Array[i] = std::min<T>(xmm0Array[i], xmm1Array[i]);

            return From(xmm2Array, 0);
        }

        [[clang::always_inline]]
        static Vec128I Max(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i16>) return _mm_max_epi16(xmm0, xmm1);

            constexpr static s00 Size = sizeof(Vec128I) / sizeof(T);

            ALIGN Array<T, Size> xmm0Array;
            ALIGN Array<T, Size> xmm1Array;
            ALIGN Array<T, Size> xmm2Array;

            Store(xmm0, xmm0Array, 0);
            Store(xmm1, xmm1Array, 0);

            for (s00 i = 0; i < Size; i++) xmm2Array[i] = std::max<T>(xmm0Array[i], xmm1Array[i]);

            return From(xmm2Array, 0);
        }

        [[clang::always_inline]]
        static Vec128I Add(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i08>) return _mm_add_epi8 (xmm0, xmm1);
            if (std::is_same_v<T, i16>) return _mm_add_epi16(xmm0, xmm1);
            if (std::is_same_v<T, i32>) return _mm_add_epi32(xmm0, xmm1);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vec128I Sub(const Vec128I& xmm0, const Vec128I& xmm1)
        {
            if (std::is_same_v<T, i08>) return _mm_sub_epi8 (xmm0, xmm1);
            if (std::is_same_v<T, i16>) return _mm_sub_epi16(xmm0, xmm1);
            if (std::is_same_v<T, i32>) return _mm_sub_epi32(xmm0, xmm1);

            __builtin_unreachable();
        }

        [[clang::always_inline]]
        static Vec128I Madd(const Vec128I& xmm0, const Vec128I& xmm1) requires std::is_same_v<T, i16>
        {
            return _mm_madd_epi16(xmm0, xmm1);
        }

        [[clang::always_inline]]
        static T Sum(const Vec128I& xmm0) requires std::is_same_v<T, i32>
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
            xmm1 = Add(xmm0, xmm1);

            ALIGN Array<T, 4> xmm1Array;

            Store(xmm1, xmm1Array, 0);

            // xmm1[0] + xmm1[1] = (a + c) + (b + d) = a + c + b + d = a + b + c + d
            return xmm1Array[0] + xmm1Array[1];
        }

    };

} // MantaRay

#endif //MANTARAY_SSE2_H

#endif
