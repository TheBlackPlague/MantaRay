//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYSUB_H
#define MANTARAY_ARRAYSUB_H

#include "../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N>
    inline void ArraySub(Array<T, N>& base, const Array<T, N>& delta)
    {
#ifdef SIMD

#ifdef __ARM_NEON__
#else

        SIMDVEC v0;
        SIMDVEC v1;

        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(base , i);
            v1 = SIMD<T>::From(delta, i);

            v0 = SIMD<T>::Sub(v0, v1);

            SIMD<T>::Store(v0, base, i);
        }

#endif

#else

        for (s00 i = 0; i < N; i++) base[i] -= delta[i];

#endif
    }

} // MantaRay

#endif //MANTARAY_ARRAYSUB_H
