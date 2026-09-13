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
    [[clang::noinline]]
    Array<U, M> ActivateFlattenAndForward(
        const Array<T, N               >& x0,
        const Array<T, N               >& x1,
        const NArray<T, M, N * 2       >& w ,
        const Array<T,               M>& b )
    {
        ALIGN Array<U, M> y;

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
            static_assert(N >= Step && N % Step == 0, "Input size must be a multiple of the SIMD width.");

            for (s00 j = 0; j < N; j += Step) {
                v2 = SIMD<T>::From(x0  , j);
                v3 = SIMD<T>::From(w[i], j);

                v2 = ActivationFunction(v2);

                v1 = SIMD<T>::Madd(v2, v3);
                v0 = SIMD<U>:: Add(v0, v1);
            }

            for (s00 j = 0; j < N; j += Step) {
                v2 = SIMD<T>::From(x1  ,     j);
                v3 = SIMD<T>::From(w[i], N + j);

                v2 = ActivationFunction(v2);

                v1 = SIMD<T>::Madd(v2, v3);
                v0 = SIMD<U>:: Add(v0, v1);
            }

            y[i] = WrapAdd(SIMD<U>::Sum(v0), static_cast<U>(b[i]));

#else

            U v0 = 0;

            for (s00 j = 0; j < N; j++) {
                const U x0Activated = static_cast<U>(ActivationFunction(x0[j]));
                const U x1Activated = static_cast<U>(ActivationFunction(x1[j]));
                const U w0 = static_cast<U>(w[i][j    ]);
                const U w1 = static_cast<U>(w[i][j + N]);

                v0 = WrapAdd(v0, WrapMul(x0Activated, w0));
                v0 = WrapAdd(v0, WrapMul(x1Activated, w1));
            }

            y[i] = WrapAdd(v0, static_cast<U>(b[i]));

#endif
        }

        return y;
    }

}

#endif //MANTARAY_ACTIVATEFLATTENANDFORWARD_H
