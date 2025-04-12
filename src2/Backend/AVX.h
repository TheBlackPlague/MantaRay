//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_AVX_H
#define MANTARAY_AVX_H

#ifdef __AVX__

#include "SSE41.h"

namespace MantaRay
{

    // 256-bit integer register
    using Vec256I = __m256i;

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

    };

} // MantaRay

#endif

#endif //MANTARAY_AVX_H
