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

            VectorE v0 = SIMD<T>::Zero;
            Vector  v1;
            Vector  v2;

            constexpr s00 Step = sizeof(Vector) / sizeof(T);

            std::array<VectorE, N / Step> v;

            for (s00 j = 0; j < N; j += Step) {
                v1 = SIMD<T>::From(x0,          j);
                v2 = SIMD<T>::From(w , stride + j);

                v1 = ActivationFunction(v1);

                v[j / Step] = SIMD<T>::Madd(v1, v2);
            }

            for (s00 j = 1; j < N / Step;       j <<= 1)
            for (s00 k = j; k < N / Step; k += (j <<  1))
                v[k - j] = SIMD<U>::Add(v[k - j], v[k]);

            v0 = SIMD<T>::Add(v0, v[0]);

            stride += N;

            for (s00 j = 0; j < N; j += Step) {
                v1 = SIMD<T>::From(x1,          j);
                v2 = SIMD<T>::From(w , stride + j);

                v1 = ActivationFunction(v1);

                v[j / Step] = SIMD<T>::Madd(v1, v2);
            }

            for (s00 j = 1; j < N / Step;       j <<= 1)
            for (s00 k = j; k < N / Step; k += (j <<  1))
                v[k - j] = SIMD<U>::Add(v[k - j], v[k]);

            v0 = SIMD<T>::Add(v0, v[0]);

            stride += N;

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
