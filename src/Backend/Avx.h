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

    };

} // Cerebrum

#endif //CEREBRUM_AVX_H
