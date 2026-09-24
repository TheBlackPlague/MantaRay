//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BACKEND_KERNEL_ACTIVATECONCATAFFINE_H
#define MANTARAY_BACKEND_KERNEL_ACTIVATECONCATAFFINE_H

#include "../Activation.h"

namespace MantaRay::Backend::Kernel
{

    template<typename A, typename Q, s00 N, s00 M>
    [[clang::noinline]]
    Array<typename Q::SumType, M> ActivateConcatAffine(
        const  Array<typename Q::FeatureType,        N>& x0     ,
        const  Array<typename Q::FeatureType,        N>& x1     ,
        const NArray<typename Q:: WeightType, M, 2 * N>& weights,
        const  Array<typename Q::    SumType,        M>& bias
    )
    {
        using F = Q::FeatureType;
        using W = Q:: WeightType;
        using S = Q::    SumType;

        using Act = Activation<A, Q>;

        ALIGN Array<S, M> output;

        for (s00 i = 0; i < M; i++) {
            constexpr s00 Step = sizeof(SIMDVEC) / sizeof(F);

            if constexpr (
                HasSIMD                &&
                std::is_same_v<F, i16> &&
                std::is_same_v<W, i16> &&
                std::is_same_v<S, i32> &&
                Act::SIMDCompatible    &&
                N >= Step && N % Step == 0
            ) {
                SIMDVEC sum = SIMD<S>::Zero;

                for (s00 j = 0; j < N; j += Step) {
                    const auto x = Act::ApplyVector(SIMD<F>::From(x0, j));

                    sum = SIMD<S>::Add(sum, SIMD<F>::Madd(x, SIMD<W>::From(weights[i],     j)));
                }

                for (s00 j = 0; j < N; j += Step) {
                    const auto x = Act::ApplyVector(SIMD<F>::From(x1, j));

                    sum = SIMD<S>::Add(sum, SIMD<F>::Madd(x, SIMD<W>::From(weights[i], N + j)));
                }

                output[i] = WrapAdd(SIMD<S>::Sum(sum), bias[i]);
            } else {
                S sum = 0;

                for (s00 j = 0; j < N; j++) {
                    sum = WrapAdd(sum, WrapMul(static_cast<S>(Act::Apply(x0[j])), static_cast<S>(weights[i][    j])));
                    sum = WrapAdd(sum, WrapMul(static_cast<S>(Act::Apply(x1[j])), static_cast<S>(weights[i][N + j])));
                }

                output[i] = WrapAdd(sum, bias[i]);
            }
        }

        return output;
    }

}

#endif
