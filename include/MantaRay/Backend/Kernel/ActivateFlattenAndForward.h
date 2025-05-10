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
            VectorE v2 = SIMD<U>::Zero;
            Vector  v3;
            Vector  v4;
            Vector  v5;
            Vector  v6;

            constexpr s00 Step = sizeof(Vector) / sizeof(T);

            for (s00 j = 0; j < N; j += Step) {
                v3 = SIMD<T>::From(x0,          j    );
                v4 = SIMD<T>::From(x1,          j    );
                v5 = SIMD<T>::From(w , stride + j    );
                v6 = SIMD<T>::From(w , stride + j + N);

                v3 = ActivationFunction(v3);
                v4 = ActivationFunction(v4);

                v1 = SIMD<T>::Madd(v3, v5);
                v2 = SIMD<T>::Madd(v4, v6);

                v0 = SIMD<U>::Add(v0, v1);
                v0 = SIMD<U>::Add(v0, v2);
            }

            stride += N * 2;

            y[i] = SIMD<U>::Sum(v0) + b[i];

#else

            U v0 = 0;

            for (s00 j = 0; j < N; j++) {
                v0 += ActivationFunction(x0[j]) * w[stride + j    ];
                v0 += ActivationFunction(x1[j]) * w[stride + j + N];
            }

            stride += N * 2;

            y[i] = v0 + b[i];

#endif
        }

        return y;
    }

}

#endif //MANTARAY_ACTIVATEFLATTENANDFORWARD_H
