//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CLIPPEDRELU_H
#define MANTARAY_CLIPPEDRELU_H

#include "../../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, T Minimum, T Maximum>
    class ClippedReLU
    {

#ifdef SIMD

#ifdef __ARM_NEON__

        using Vector = SIMDVEC<T>;

#else

        using Vector = SIMDVEC;

#endif

        public:
        [[clang::always_inline]]
        static Vector Activate(const Vector& value)
        {
            const Vector Min = SIMD<T>::From(Minimum);
            const Vector Max = SIMD<T>::From(Maximum);

            return SIMD<T>::Min(Max, SIMD<T>::Max(Min, value));
        }

#else

        public:
        [[clang::always_inline]]
        static T Activate(const T value) { return std::min(Maximum, std::max(Minimum, value)); }

#endif

    };

} // MantaRay

#endif //MANTARAY_CLIPPEDRELU_H
