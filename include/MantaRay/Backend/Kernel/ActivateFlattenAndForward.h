//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACTIVATEFLATTENANDFORWARD_H
#define MANTARAY_ACTIVATEFLATTENANDFORWARD_H

#include "../Processor.h"

namespace MantaRay
{

    template<auto ActivationFunction, QuantizedInteger T, QuantizedInteger U, s00 N, s00 M>
    Array<U, M> ActivateFlattenAndForward(
        const Array<T, N        >& x0,
        const Array<T, N        >& x1,
        const Array<T, N * 2 * M>& w ,
        const Array<T,         M>& b )
    {
        ALIGN Array<U, M> y;

        s00 stride = 0;

        for (s00 i = 0; i < M; i++) {
#ifdef SIMD

#ifdef __ARM_NEON__

            using Vector  = SIMDVEC       <T>;
            using VectorE = SIMDVEC_EXTEND<T>;

#else

            using Vector  = SIMDVEC;
            using VectorE = SIMDVEC;

#endif

            VectorE v0 = SIMD<U>::Zero;
            VectorE v1 = SIMD<U>::Zero;
            Vector  v2;
            Vector  v3;

            constexpr s00 Step = sizeof(Vector) / sizeof(T);

            for (s00 j = 0; j < N; j += Step) {
                v2 = SIMD<T>::From(x0,          j);
                v3 = SIMD<T>::From(w , stride + j);

                v2 = ActivationFunction(v2);

                v1 = SIMD<T>::Madd(v2, v3);
                v0 = SIMD<U>:: Add(v0, v1);
            }

            stride += N;

            for (s00 j = 0; j < N; j += Step) {
                v2 = SIMD<T>::From(x1,          j);
                v3 = SIMD<T>::From(w , stride + j);

                v2 = ActivationFunction(v2);

                v1 = SIMD<T>::Madd(v2, v3);
                v0 = SIMD<U>:: Add(v0, v1);
            }

            stride += N;

            y[i] = SIMD<U>::Sum(v0) + b[i];

#else

            T v0 = 0;

            for (s00 j = 0; j < N; j++) v0 += ActivationFunction(x[j] * w[stride + j]);

            stride += N;

            y[i] = v0 + b[i];

#endif
        }

        return y;
    }

}

#endif //MANTARAY_ACTIVATEFLATTENANDFORWARD_H
