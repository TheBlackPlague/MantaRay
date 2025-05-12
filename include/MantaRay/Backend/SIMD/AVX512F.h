//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __AVX512F__

#ifndef MANTARAY_AVX512F_H
#define MANTARAY_AVX512F_H

#include "AVX2.h"

namespace MantaRay
{

    // 512-bit integer register
    using Vec512I = __m512i;

#ifdef ALIGN
#undef ALIGN
#endif

#define ALIGN alignas(sizeof(MantaRay::Vec512I))

    template<QuantizedInteger T>
    struct AVX512F : AVX2<T>
    {

        constexpr static Vec512I Zero = _mm512_setzero_si512();

        [[clang::always_inline]]
        static Vec512I From(const T value)
        {
            if (std::is_same_v<T, i08>) return _mm512_set1_epi8 (value);
            if (std::is_same_v<T, i16>) return _mm512_set1_epi16(value);
            if (std::is_same_v<T, i32>) return _mm512_set1_epi32(value);

            __builtin_unreachable();
        }

        template<s00 Size>
        [[clang::always_inline]]
        static Vec512I From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec512I), "Array size must be at least the width of a Vec512I.");

            return _mm512_load_si512(reinterpret_cast<Vec512I const*>(&array[index]));
        }

        template<s00 Size>
        [[clang::always_inline]]
        static void Store(const Vec512I& zmm0, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec512I), "Array size must be at least the width of a Vec512I.");

            _mm512_store_si512(reinterpret_cast<Vec512I*>(&array[index]), zmm0);
        }

        [[clang::always_inline]]
        static Vec512I Min(const Vec512I& zmm0, const Vec512I& zmm1)
        {
            if (std::is_same_v<T, i32>) return _mm512_min_epi32(zmm0, zmm1);

            // zmm0 = [a00, a01, a02, a03, ..., a28, a29, a30, a31] -- Assuming T is i16
            // zmm1 = [b00, b01, b02, b03, ..., b28, b29, b30, b31] -- Assuming T is i16

            Vec256I ymm0;
            Vec256I ymm1;
            Vec256I ymm2;
            Vec256I ymm3;

            Vec512I zmm2;

            // ymm0 = [a00, a01, a02, a03, ..., a12, a13, a14, a15]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [b00, b01, b02, b03, ..., b12, b13, b14, b15]
            ymm1 = _mm512_extracti64x4_epi64(zmm1, 0);

            // ymm2 = [a16, a17, a18, a19, ..., a28, a29, a30, a31]
            ymm2 = _mm512_extracti64x4_epi64(zmm0, 1);

            // ymm3 = [b16, b17, b18, b19, ..., b28, b29, b30, b31]
            ymm3 = _mm512_extracti64x4_epi64(zmm1, 1);

            // ymm0 = [min(a00, b00), min(a01, b01), ..., min(a14, b14), min(a15, b15)]
            ymm0 = AVX2<T>::Min(ymm0, ymm1);

            // ymm1 = [min(a16, b16), min(a17, b17), ..., min(a30, b30), min(a31, b31)]
            ymm1 = AVX2<T>::Min(ymm2, ymm3);

            // zmm2 = [min(a00, b00), min(a01, b01), ..., min(a14, b14), min(a15, b15), 0, 0, ..., 0, 0]
            zmm2 = _mm512_inserti64x4(Zero, ymm0, 0);

            // zmm2 = [min(a00, b00), min(a01, b01), ..., min(a14, b14), min(a15, b15),
            //         min(a16, b16), min(a17, b17), ..., min(a30, b30), min(a31, b31)]
            // zmm2 = [min(a00, b00), min(a01, b01), ..., min(a30, b30), min(a31, b31)]
            zmm2 = _mm512_inserti64x4(zmm2, ymm1, 1);

            return zmm2;
        }

        [[clang::always_inline]]
        static Vec512I Max(const Vec512I& zmm0, const Vec512I& zmm1)
        {
            if (std::is_same_v<T, i32>) return _mm512_max_epi32(zmm0, zmm1);

            // zmm0 = [a00, a01, a02, a03, ..., a28, a29, a30, a31] -- Assuming T is i16
            // zmm1 = [b00, b01, b02, b03, ..., b28, b29, b30, b31] -- Assuming T is i16

            Vec256I ymm0;
            Vec256I ymm1;
            Vec256I ymm2;
            Vec256I ymm3;

            Vec512I zmm2;

            // ymm0 = [a00, a01, a02, a03, ..., a12, a13, a14, a15]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [b00, b01, b02, b03, ..., b12, b13, b14, b15]
            ymm1 = _mm512_extracti64x4_epi64(zmm1, 0);

            // ymm2 = [a16, a17, a18, a19, ..., a28, a29, a30, a31]
            ymm2 = _mm512_extracti64x4_epi64(zmm0, 1);

            // ymm3 = [b16, b17, b18, b19, ..., b28, b29, b30, b31]
            ymm3 = _mm512_extracti64x4_epi64(zmm1, 1);

            // ymm0 = [max(a00, b00), max(a01, b01), ..., max(a14, b14), max(a15, b15)]
            ymm0 = AVX2<T>::Max(ymm0, ymm1);

            // ymm1 = [max(a16, b16), max(a17, b17), ..., max(a30, b30), max(a31, b31)]
            ymm1 = AVX2<T>::Max(ymm2, ymm3);

            // zmm2 = [max(a00, b00), max(a01, b01), ..., max(a14, b14), max(a15, b15), 0, 0, ..., 0, 0]
            zmm2 = _mm512_inserti64x4(Zero, ymm0, 0);

            // zmm2 = [max(a00, b00), max(a01, b01), ..., max(a14, b14), max(a15, b15),
            //         max(a16, b16), max(a17, b17), ..., max(a30, b30), max(a31, b31)]
            // zmm2 = [max(a00, b00), max(a01, b01), ..., max(a30, b30), max(a31, b31)]
            zmm2 = _mm512_inserti64x4(zmm2, ymm1, 1);

            return zmm2;
        }

        [[clang::always_inline]]
        static Vec512I Add(const Vec512I& zmm0, const Vec512I& zmm1)
        {
            if (std::is_same_v<T, i32>) return _mm512_add_epi32(zmm0, zmm1);

            // zmm0 = [a00, a01, a02, a03, ..., a28, a29, a30, a31] -- Assuming T is i16
            // zmm1 = [b00, b01, b02, b03, ..., b28, b29, b30, b31] -- Assuming T is i16

            Vec256I ymm0;
            Vec256I ymm1;
            Vec256I ymm2;
            Vec256I ymm3;

            Vec512I zmm2;

            // ymm0 = [a00, a01, a02, a03, ..., a12, a13, a14, a15]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [b00, b01, b02, b03, ..., b12, b13, b14, b15]
            ymm1 = _mm512_extracti64x4_epi64(zmm1, 0);

            // ymm2 = [a16, a17, a18, a19, ..., a28, a29, a30, a31]
            ymm2 = _mm512_extracti64x4_epi64(zmm0, 1);

            // ymm3 = [b16, b17, b18, b19, ..., b28, b29, b30, b31]
            ymm3 = _mm512_extracti64x4_epi64(zmm1, 1);

            //     [   a00   ,    a01   , ...,    a14   ,    a15   ]
            // +   [   b00   ,    b01   , ...,    b14   ,    b15   ]
            // =   [a00 + b00, a01 + b01, ..., a14 + b14, a15 + b15]
            ymm0 = AVX2<T>::Add(ymm0, ymm1);

            //     [   a16   ,    a17   , ...,    a30   ,    a31   ]
            // +   [   b16   ,    b17   , ...,    b30   ,    b31   ]
            // =   [a16 + b16, a17 + b17, ..., a30 + b30, a31 + b31]
            ymm1 = AVX2<T>::Add(ymm2, ymm3);

            // zmm2 = [a00 + b00, a01 + b01, ..., a14 + b14, a15 + b15, 0, 0, ..., 0, 0]
            zmm2 = _mm512_inserti64x4(Zero, ymm0, 0);

            // zmm2 = [a00 + b00, a01 + b01, ..., a14 + b14, a15 + b15,
            //         a16 + b16, a17 + b17, ..., a30 + b30, a31 + b31]
            // zmm2 = [a00 + b00, a01 + b01, ..., a30 + b30, a31 + b31]
            zmm2 = _mm512_inserti64x4(zmm2, ymm1, 1);

            return zmm2;
        }

        [[clang::always_inline]]
        static Vec512I Sub(const Vec512I& zmm0, const Vec512I& zmm1)
        {
            if (std::is_same_v<T, i32>) return _mm512_sub_epi32(zmm0, zmm1);

            // zmm0 = [a00, a01, a02, a03, ..., a28, a29, a30, a31] -- Assuming T is i16
            // zmm1 = [b00, b01, b02, b03, ..., b28, b29, b30, b31] -- Assuming T is i16

            Vec256I ymm0;
            Vec256I ymm1;
            Vec256I ymm2;
            Vec256I ymm3;

            Vec512I zmm2;

            // ymm0 = [a00, a01, a02, a03, ..., a12, a13, a14, a15]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [b00, b01, b02, b03, ..., b12, b13, b14, b15]
            ymm1 = _mm512_extracti64x4_epi64(zmm1, 0);

            // ymm2 = [a16, a17, a18, a19, ..., a28, a29, a30, a31]
            ymm2 = _mm512_extracti64x4_epi64(zmm0, 1);

            // ymm3 = [b16, b17, b18, b19, ..., b28, b29, b30, b31]
            ymm3 = _mm512_extracti64x4_epi64(zmm1, 1);

            //     [   a00   ,    a01   , ...,    a14   ,    a15   ]
            // -   [   b00   ,    b01   , ...,    b14   ,    b15   ]
            // =   [a00 - b00, a01 - b01, ..., a14 - b14, a15 - b15]
            ymm0 = AVX2<T>::Sub(ymm0, ymm1);

            //     [   a16   ,    a17   , ...,    a30   ,    a31   ]
            // -   [   b16   ,    b17   , ...,    b30   ,    b31   ]
            // =   [a16 - b16, a17 - b17, ..., a30 - b30, a31 - b31]
            ymm1 = AVX2<T>::Sub(ymm2, ymm3);

            // zmm2 = [a00 - b00, a01 - b01, ..., a14 - b14, a15 - b15, 0, 0, ..., 0, 0]
            zmm2 = _mm512_inserti64x4(Zero, ymm0, 0);

            // zmm2 = [a00 - b00, a01 - b01, ..., a14 - b14, a15 - b15,
            //         a16 - b16, a17 - b17, ..., a30 - b30, a31 - b31]
            // zmm2 = [a00 - b00, a01 - b01, ..., a30 - b30, a31 - b31]
            zmm2 = _mm512_inserti64x4(zmm2, ymm1, 1);

            return zmm2;
        }

        [[clang::always_inline]]
        static Vec512I Madd(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i16>
        {
            // zmm0 = [a00, a01, a02, a03, ..., a28, a29, a30, a31]
            // zmm1 = [b00, b01, b02, b03, ..., b28, b29, b30, b31]

            Vec256I ymm0;
            Vec256I ymm1;
            Vec256I ymm2;
            Vec256I ymm3;

            Vec512I zmm2;

            // ymm0 = [a00, a01, a02, a03, ..., a12, a13, a14, a15]
            ymm0 = _mm512_extracti64x4_epi64(zmm0, 0);

            // ymm1 = [b00, b01, b02, b03, ..., b12, b13, b14, b15]
            ymm1 = _mm512_extracti64x4_epi64(zmm1, 0);

            // ymm2 = [a16, a17, a18, a19, ..., a28, a29, a30, a31]
            ymm2 = _mm512_extracti64x4_epi64(zmm0, 1);

            // ymm3 = [b16, b17, b18, b19, ..., b28, b29, b30, b31]
            ymm3 = _mm512_extracti64x4_epi64(zmm1, 1);

            //     [   a00    ,    a01   , ...,    a14    ,    a15   ]
            // *   [   b00    ,    b01   , ...,    b14    ,    b15   ]
            // =   [a00 * b00 , a01 * b01, ..., a14 * b14 , a15 * b15] -- INTERMEDIATE
            // =   [a00 * b00 + a01 * b01, ..., a14 * b14 + a15 * b15] -- FINAL
            ymm0 = AVX2<T>::Madd(ymm0, ymm1);

            //     [   a16    ,    a17   , ...,    a30    ,    a31   ]
            // *   [   b16    ,    b17   , ...,    b30    ,    b31   ]
            // =   [a16 * b16 , a17 * b17, ..., a30 * b30 , a31 * b31] -- INTERMEDIATE
            // =   [a16 * b16 + a17 * b17, ..., a30 * b30 + a31 * b31] -- FINAL
            ymm1 = AVX2<T>::Madd(ymm2, ymm3);

            // zmm2 = [a00 * b00 + a01 * b01, ..., a14 * b14 + a15 * b15, 0, 0, ..., 0, 0]
            zmm2 = _mm512_inserti64x4(Zero, ymm0, 0);

            // zmm2 = [a00 * b00 + a01 * b01, ..., a14 * b14 + a15 * b15,
            //         a16 * b16 + a17 * b17, ..., a30 * b30 + a31 * b31]
            zmm2 = _mm512_inserti64x4(zmm2, ymm1, 1);

            return zmm2;
        }

        [[clang::always_inline]]
        static T Sum(const Vec512I& zmm0) requires std::is_same_v<T, i32>
        {
            return _mm512_reduce_add_epi32(zmm0);
        }

    };

} // MantaRay

#endif //MANTARAY_AVX512F_H

#endif
