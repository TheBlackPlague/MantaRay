//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_AVX512_H
#define CEREBRUM_AVX512_H

#include "Avx2.h"

namespace Cerebrum
{

    template<typename T>
    class Avx512
    {

        static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>,
                      "Unsupported type provided.");

        public:
            static inline Vec512I Zero()
            {
                return _mm512_setzero_si512();
            }

            static inline Vec512I From(const T value)
            {
                if (std::is_same_v<T, int8_t> ) return _mm512_set1_epi8 (value);

                if (std::is_same_v<T, int16_t>) return _mm512_set1_epi16(value);

                if (std::is_same_v<T, int32_t>) return _mm512_set1_epi32(value);
            }

            template<size_t Size>
            static inline Vec512I From(const std::array<T, Size> &array, const uint32_t index)
            {
                return _mm512_load_si512((Vec512I const*) &array[index]);
            }

            template<size_t Size>
            static inline void Store(const Vec256I &zmm0, std::array<T, Size> &array, const uint32_t index)
            {
                _mm512_store_si512((Vec512I *) &array[index], zmm0);
            }

            static inline Vec512I Min(const Vec512I &zmm0, const Vec512I &zmm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm512_min_epi8 (zmm0, zmm1);

                if (std::is_same_v<T, int16_t>) return _mm512_min_epi16(zmm0, zmm1);

                if (std::is_same_v<T, int32_t>) return _mm512_min_epi32(zmm0, zmm1);
            }

            static inline Vec512I Max(const Vec512I &zmm0, const Vec512I &zmm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm512_max_epi8 (zmm0, zmm1);

                if (std::is_same_v<T, int16_t>) return _mm512_max_epi16(zmm0, zmm1);

                if (std::is_same_v<T, int32_t>) return _mm512_max_epi32(zmm0, zmm1);
            }

            static inline Vec512I Add(const Vec512I &zmm0, const Vec512I &zmm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm512_add_epi8 (zmm0, zmm1);

                if (std::is_same_v<T, int16_t>) return _mm512_add_epi16(zmm0, zmm1);

                if (std::is_same_v<T, int32_t>) return _mm512_add_epi32(zmm0, zmm1);
            }

            static inline Vec512I Subtract(const Vec512I &zmm0, const Vec512I &zmm1)
            {
                if (std::is_same_v<T, int8_t> ) return _mm512_sub_epi8 (zmm0, zmm1);

                if (std::is_same_v<T, int16_t>) return _mm512_sub_epi16(zmm0, zmm1);

                if (std::is_same_v<T, int32_t>) return _mm512_sub_epi32(zmm0, zmm1);
            }

            static inline Vec512I MultiplyAndAddAdjacent(const Vec512I &zmm0, const Vec512I &zmm1)
            {
                static_assert(std::is_same_v<T, int16_t>, "Unsupported type provided.");

                return _mm512_madd_epi16(zmm0, zmm1);
            }

    };

} // Cerebrum

#endif //CEREBRUM_AVX512_H
