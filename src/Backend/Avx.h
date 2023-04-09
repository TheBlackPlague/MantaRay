//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_AVX_H
#define CEREBRUM_AVX_H

#include <xtr1common>
#include <cstdint>
#include <array>
#include "RegisterDefinition.h"

namespace Cerebrum
{

    template<typename T>
    class Avx
    {

        static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>,
                "Unsupported type provided.");

        public:
            static inline Vec256I Zero()
            {
                return _mm256_setzero_si256();
            }

            static inline Vec256I From(const T value)
            {
                if (std::is_same_v<T, int8_t> ) return _mm256_set1_epi8 (value);

                if (std::is_same_v<T, int16_t>) return _mm256_set1_epi16(value);

                if (std::is_same_v<T, int32_t>) return _mm256_set1_epi32(value);
            }

            template<size_t Size>
            static inline Vec256I From(const std::array<T, Size> &array, const uint32_t index)
            {
                return _mm256_load_si256((Vec256I const*) &array[index]);
            }

            static inline Vec256I Min(const Vec256I &ymm0, const Vec256I &ymm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm256_min_epi8 (ymm0, ymm1);

                if (std::is_same_v<T, int16_t>) return _mm256_min_epi16(ymm0, ymm1);

                if (std::is_same_v<T, int32_t>) return _mm256_min_epi32(ymm0, ymm1);
            }

            static inline Vec256I Max(const Vec256I &ymm0, const Vec256I &ymm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm256_max_epi8 (ymm0, ymm1);

                if (std::is_same_v<T, int16_t>) return _mm256_max_epi16(ymm0, ymm1);

                if (std::is_same_v<T, int32_t>) return _mm256_max_epi32(ymm0, ymm1);
            }

            static inline Vec256I Add(const Vec256I &ymm0, const Vec256I &ymm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm256_add_epi8 (ymm0, ymm1);

                if (std::is_same_v<T, int16_t>) return _mm256_add_epi16(ymm0, ymm1);

                if (std::is_same_v<T, int32_t>) return _mm256_add_epi32(ymm0, ymm1);
            }

            static inline T Sum(const Vec256I &ymm0)
            {
                static_assert(std::is_same_v<T, int32_t>, "Unsupported type provided.");

                Vec128I xmm0;
                Vec128I xmm1;

                xmm0 = _mm256_castsi256_si128(ymm0);
                xmm1 = _mm256_extracti128_si256(ymm0, 1);
                xmm0 = _mm_add_epi32(xmm0, xmm1);
                xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);
                xmm0 = _mm_add_epi32(xmm0, xmm1);
                xmm1 = _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(2, 3, 0, 1));
                xmm0 = _mm_add_epi32(xmm0, xmm1);
                return _mm_cvtsi128_si32(xmm0);
            }

    };

} // Cerebrum

#endif //CEREBRUM_AVX_H
