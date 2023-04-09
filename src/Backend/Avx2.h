//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_AVX2_H
#define CEREBRUM_AVX2_H

#include "Avx.h"

namespace Cerebrum
{

    template<typename T>
    class Avx2
    {

        public:
            static inline Vec256I MultiplyAndAddAdjacent(const Vec256I &ymm0, const Vec256I &ymm1)
            {
                return _mm256_madd_epi16(ymm0, ymm1);
            }

    };

} // Cerebrum

#endif //CEREBRUM_AVX2_H
