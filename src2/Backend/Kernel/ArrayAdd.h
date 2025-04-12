//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYADD_H
#define MANTARAY_ARRAYADD_H

#include "../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N>
    static inline void ArrayAdd(Array<T, N>& base, const Array<T, N>& delta)
    {
#ifdef SIMD

#ifdef __ARM_NEON__

        SIMDVEC<T> v0;
        SIMDVEC<T> v1;

        constexpr static s00 Step = sizeof(SIMDVEC<T>) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(base , i);
            v1 = SIMD<T>::From(delta, i);

            v0 = SIMD<T>::Add(v0, v1);

            SIMD<T>::Store(v0, base, i);
        }

#else

        SIMDVEC v0;
        SIMDVEC v1;

        constexpr static s00 Step = sizeof(SIMDVEC) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(base , i);
            v1 = SIMD<T>::From(delta, i);

            v0 = SIMD<T>::Add(v0, v1);

            SIMD<T>::Store(v0, base, i);
        }

#endif

#else

        for (s00 i = 0; i < N; i++) base[i] += delta[i];

#endif
    }

} // MantaRay

#endif //MANTARAY_ARRAYADD_H
