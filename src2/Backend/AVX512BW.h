//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_AVX512BW_H
#define MANTARAY_AVX512BW_H

#ifdef __AVX512BW__

#include "AVX512F.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    struct AVX512BW : AVX512F<T>
    {

        static inline Vec512I Min(const Vec512I& ymm0, const Vec512I& ymm1)
        requires std::is_same_v<T, i08> || std::is_same_v<T, i16>
        {
            if (std::is_same_v<T, i08>) return _mm512_min_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm512_min_epi16(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec512I Max(const Vec512I& ymm0, const Vec512I& ymm1)
        requires std::is_same_v<T, i08> || std::is_same_v<T, i16>
        {
            if (std::is_same_v<T, i08>) return _mm512_max_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm512_max_epi16(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec512I Add(const Vec512I& ymm0, const Vec512I& ymm1)
        requires std::is_same_v<T, i08> || std::is_same_v<T, i16>
        {
            if (std::is_same_v<T, i08>) return _mm512_add_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm512_add_epi16(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec512I Sub(const Vec512I& ymm0, const Vec512I& ymm1)
        requires std::is_same_v<T, i08> || std::is_same_v<T, i16>
        {
            if (std::is_same_v<T, i08>) return _mm512_sub_epi8 (ymm0, ymm1);
            if (std::is_same_v<T, i16>) return _mm512_sub_epi16(ymm0, ymm1);

            __builtin_unreachable();
        }

        static inline Vec512I Madd(const Vec512I& zmm0, const Vec512I& zmm1) requires std::is_same_v<T, i16>
        {
            return _mm512_madd_epi16(zmm0, zmm1);
        }

    };

} // MantaRay

#endif

#endif //MANTARAY_AVX512BW_H
