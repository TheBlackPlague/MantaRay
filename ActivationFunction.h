//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_ACTIVATIONFUNCTION_H
#define CEREBRUM_ACTIVATIONFUNCTION_H

#include <vector>
#include <immintrin.h>
#include "SIMDBackend.h"

namespace Cerebrum
{

    template<typename Type, Type Min, Type Max>
    class ClippedReLU
    {
        public:
#ifdef __AVX__
            inline static __m256i Activate(const __m256i &ymm0)
            {
                const __m256i& ymm1 = SIMDBackend<Type>::From(Min);
                const __m256i& ymm2 = SIMDBackend<Type>::From(Max);

                return SIMDBackend<Type>::Max(ymm1, SIMDBackend<Type>::Min(ymm2, ymm0));
            }
#else
            inline static Type Activate(const Type arg)
            {
                return std::max(Min, std::min(Max, arg));
            }
#endif

    };

} // Cerebrum

#endif //CEREBRUM_ACTIVATIONFUNCTION_H
