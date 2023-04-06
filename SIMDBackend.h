//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_SIMDBACKEND_H
#define CEREBRUM_SIMDBACKEND_H

#include <immintrin.h>
#include <xtr1common>
#include <cstdint>

namespace Cerebrum
{

    template<typename Type>
    class SIMDBackend
    {

        static_assert(std::is_same_v<Type, int8_t> || std::is_same_v<Type, int16_t>
                      || std::is_same_v<Type, int32_t>, "Unsupported type provided.");

        public:
#ifdef __AVX__
            inline static __m256i From(const Type value)
            {
                if constexpr(std::is_same_v<Type, int8_t>) return _mm256_set1_epi8(value);
                else if (std::is_same_v<Type, int16_t>) return _mm256_set1_epi16(value);
                else return _mm256_set1_epi32(value);
            }

            template<size_t Size>
            inline static __m256i From(const std::array<Type, Size> &valueArray, const int index)
            {
                return _mm256_load_si256((__m256i const*) &valueArray[index]);
            }

            inline static __m256i Zero()
            {
                return _mm256_setzero_si256();
            }

            inline static __m256i Min(const __m256i &ymm0, const __m256i &ymm1)
            {
                if constexpr(std::is_same_v<Type, int8_t>) return _mm256_min_epi8(ymm0, ymm1);
                else if (std::is_same_v<Type, int16_t>) return _mm256_min_epi16(ymm0, ymm1);
                else return _mm256_min_epi32(ymm0, ymm1);
            }

            inline static __m256i Max(const __m256i &ymm0, const __m256i &ymm1)
            {
                if constexpr(std::is_same_v<Type, int8_t>) return _mm256_max_epi8(ymm0, ymm1);
                else if (std::is_same_v<Type, int16_t>) return _mm256_max_epi16(ymm0, ymm1);
                else return _mm256_max_epi32(ymm0, ymm1);
            }

            inline static __m256i Add(const __m256i &ymm0, const __m256i &ymm1)
            {
                if constexpr(std::is_same_v<Type, int8_t>) return _mm256_add_epi8(ymm0, ymm1);
                else if (std::is_same_v<Type, int16_t>) return _mm256_add_epi16(ymm0, ymm1);
                else return _mm256_add_epi32(ymm0, ymm1);
            }

#ifdef __AVX2__
            inline static __m256i MultiplyAddAdjacent(const __m256i &ymm0, const __m256i &ymm1)
            {
                static_assert(std::is_same_v<Type, int16_t>, "Unsupported type provided.");

                return _mm256_madd_epi16(ymm0, ymm1);
            }
#endif

            inline static Type Sum(const __m256i &ymm0)
            {
                static_assert(std::is_same_v<Type, int32_t>, "Unsupported type provided.");

                __m128i xmm0 = _mm_add_epi32(_mm256_castsi256_si128(ymm0), _mm256_extracti128_si256(ymm0, 1));
                __m128i xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);
                __m128i xmm2 = _mm_add_epi32(xmm1, xmm0);
                __m128i xmm3 = _mm_shuffle_epi32(xmm2, _MM_SHUFFLE(2, 3, 0, 1));
                __m128i xmm4 = _mm_add_epi32(xmm2, xmm3);
                return _mm_cvtsi128_si32(xmm4);

//                __m128i xmm0 = _mm_add_epi32(_mm256_extractf128_si256(ymm0, 0), _mm256_extractf128_si256(ymm0, 1));
//                __m128i xmm1 = _mm_add_epi32(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(2, 3, 0, 1)));
//                __m128i xmm2 = _mm_add_epi32(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(1, 0, 3, 2)));
//                return _mm_cvtsi128_si32(xmm2);
            }
#endif

    };

} // Cerebrum

#endif //CEREBRUM_SIMDBACKEND_H
