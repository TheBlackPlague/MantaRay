//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_FUSED_ACTIVATECONCATAFFINE_H
#define MANTARAY_BACKEND_KERNEL_FUSED_ACTIVATECONCATAFFINE_H

#include <memory>

#include "../Activation.h"
#include "../../../Common/Alignment.h"
#include "../../../Common/Container.h"

namespace MantaRay::Backend::Kernel
{

    template<typename A, typename Q, s00 N, s00 M, bool AllowNarrowSquare = true, bool Aligned = false>
    [[clang::noinline]]
    Array<typename Q::SumType, M> ActivateConcatAffine(
        const  Array<typename Q::FeatureType,        N>& first  ,
        const  Array<typename Q::FeatureType,        N>& second ,
        const NArray<typename Q:: WeightType, M, 2 * N>& weights,
        const  Array<typename Q::    SumType,        M>& bias
    )
    {
        using F = Q::FeatureType;
        using W = Q:: WeightType;
        using S = Q::    SumType;

        using Act = Activation<A, Q>;

        constexpr s00 Lanes = Native::NativeLanes<F>;

        constexpr bool Proven = !requires { typename Act::Proof; } || AllowNarrowSquare;

        constexpr bool Vectorized = Proven && std::is_same_v<F, i16> && std::is_same_v<W, i16> &&
            std::is_same_v<S, i32> && Act::SIMDCompatible && Lanes > 1 && N % Lanes == 0;

        ALIGN Array<S, M> result;

        for (s00 output = 0; output < M; output++) {
            if constexpr (Vectorized) {
                auto sums = Native::Broadcast<S, Lanes / 2>(0);

                const F*  firstValues =  first.data();
                const F* secondValues = second.data();

                const W* columns = weights[output].data();

                if (Aligned) {
                     firstValues = std::assume_aligned<Alignment>( firstValues);
                    secondValues = std::assume_aligned<Alignment>(secondValues);

                    columns = std::assume_aligned<Alignment>(columns);
                }

                for (s00 input = 0; input < N; input += Lanes) {
                    const auto values = Act::ApplyVector(Native::Load<F, Lanes>( firstValues + input));
                    const auto column = Native::Load<W, Lanes>(columns + input);

                    sums = Native::Add(sums, Native::MultiplyAddPairs(values, column));
                }

                for (s00 input = 0; input < N; input += Lanes) {
                    const auto values = Act::ApplyVector(Native::Load<F, Lanes>(secondValues + input));
                    const auto column = Native::Load<W, Lanes>(columns + N + input);

                    sums = Native::Add(sums, Native::MultiplyAddPairs(values, column));
                }

                result[output] = WrapAdd(Native::Sum(sums), bias[output]);
            } else {
                S sum = 0;

                for (s00 input = 0; input < N; input++) {
                    const S a = static_cast<S>(Act::Apply( first[input]));
                    const S b = static_cast<S>(Act::Apply(second[input]));

                    const S x = static_cast<S>(weights[output][    input]);
                    const S y = static_cast<S>(weights[output][N + input]);

                    sum = WrapAdd(sum, WrapMul(a, x));
                    sum = WrapAdd(sum, WrapMul(b, y));
                }

                result[output] = WrapAdd(sum, bias[output]);
            }
        }

        return result;
    }

}

#endif
