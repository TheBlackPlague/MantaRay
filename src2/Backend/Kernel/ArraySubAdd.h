//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYSUBADD_H
#define MANTARAY_ARRAYSUBADD_H

#include "../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N>
    inline void ArraySubAdd(Array<T, N>& base, const Array<T, N>& sub, const Array<T, N>& add)
    {
#ifdef SIMD

#ifdef __ARM_NEON__

        SIMDVEC<T> v0;
        SIMDVEC<T> v1;
        SIMDVEC<T> v2;

        constexpr s00 Step = sizeof(SIMDVEC<T>) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(base, i);
            v1 = SIMD<T>::From(sub , i);
            v2 = SIMD<T>::From(add , i);

            v0 = SIMD<T>::Sub(v0, v1);
            v0 = SIMD<T>::Add(v0, v2);

            SIMD<T>::Store(v0, base, i);
        }

#else

        SIMDVEC v0;
        SIMDVEC v1;
        SIMDVEC v2;

        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(base, i);
            v1 = SIMD<T>::From(sub , i);
            v2 = SIMD<T>::From(add , i);

            v0 = SIMD<T>::Sub(v0, v1);
            v0 = SIMD<T>::Add(v0, v2);

            SIMD<T>::Store(v0, base, i);
        }

#endif

#else

        for (s00 i = 0; i < N; i++) base[i] = base[i] - sub[i] + add[i];

#endif
    }

}

#endif //MANTARAY_ARRAYSUBADD_H
