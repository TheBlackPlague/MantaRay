//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACTIVATEFLATTENANDFORWARD_H
#define MANTARAY_ACTIVATEFLATTENANDFORWARD_H

#include "../Processor.h"

namespace MantaRay
{

    constexpr u64 CeilSqrt(const u64 n)
    {
        u64 l = 1, h = 1ULL << 32;
        while (l < h) {
            const u64 m = (l + h) >> 1;

            if (m * m < n) l = m + 1;
            else           h = m    ;
        }

        return l;
    }

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

            constexpr s00 Lanes = 3;

            VectorE v32[Lanes * 2];
            Vector  v16[Lanes * 2];

            for (s00 r = Lanes; r < Lanes * 2; r++) v32[r] = SIMD<U>::Zero;

            constexpr s00 Step = sizeof(Vector) / sizeof(T);

            auto Inner = [&w, &v32, &v16](const Array<T, N>& x, const s00 s, const s00 j) -> void
            {
                UNROLL
                for (s00 r = 0; r < Lanes; r++) v16[        r] = SIMD<T>::From(x,     j + Step * r);
                UNROLL
                for (s00 r = 0; r < Lanes; r++) v16[Lanes + r] = SIMD<T>::From(w, s + j + Step * r);

                UNROLL
                for (s00 r = 0; r < Lanes; r++) v16[        r] = ActivationFunction(v16[r]);

                UNROLL
                for (s00 r = 0; r < Lanes; r++) v32[        r] = SIMD<T>::Madd(v16[r], v16[Lanes + r]);
                UNROLL
                for (s00 r = 0; r < Lanes; r++) v32[Lanes + r] = SIMD<U>:: Add(v32[r], v32[Lanes + r]);
            };

            UNROLL
            for (s00 j = 0; j < N; j += Step * Lanes) Inner(x0, stride, j);

            stride += N;

            UNROLL
            for (s00 j = 0; j < N; j += Step * Lanes) Inner(x1, stride, j);

            stride += N;

            constexpr s00 LaneAccumulationDepth = CeilSqrt(Lanes);

            UNROLL
            for (s00 l = 0; l < LaneAccumulationDepth; l++)
            UNROLL
            for (s00 r = 0, d = 1ULL << l; r + d < Lanes; r += d * 2)
                v32[Lanes + r] = SIMD<U>::Add(v32[Lanes + r], v32[Lanes + r + d]);

            y[i] = SIMD<U>::Sum(v32[Lanes]) + b[i];

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
