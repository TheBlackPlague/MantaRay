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

                if constexpr (Native::DotAccumulators<F> > 1) {
                    constexpr s00 Count = Native::DotAccumulators<F>;

                    static_assert(Count % 2 == 0);

                    constexpr s00 Step = Lanes * (Count / 2);

                    std::array<Native::Register<S, Lanes / 2>, Count> partial;

                    for (auto& sum : partial) sum = Native::Broadcast<S, Lanes / 2>(0);

                    s00 input = 0;
                    for (; input + Step <= N; input += Step) {
                        #pragma unroll
                        for (s00 part = 0; part < Count / 2; part++) {
                            const s00 index = input + part * Lanes;

                            const auto a = Act::ApplyVector(Native::Load<F, Lanes>( firstValues + index));
                            const auto b = Act::ApplyVector(Native::Load<F, Lanes>(secondValues + index));

                            partial[2 * part    ] = Native::AccumulateDot(
                                partial[2 * part    ],
                                a,
                                Native::Load<W, Lanes>(columns +     index)
                            );
                            partial[2 * part + 1] = Native::AccumulateDot(
                                partial[2 * part + 1],
                                b,
                                Native::Load<W, Lanes>(columns + N + index)
                            );
                        }
                    }

                    for (; input < N; input += Lanes) {
                        const auto a = Act::ApplyVector(Native::Load<F, Lanes>( firstValues + input));
                        const auto b = Act::ApplyVector(Native::Load<F, Lanes>(secondValues + input));

                        partial[0] = Native::AccumulateDot(partial[0], a, Native::Load<W, Lanes>(columns +     input));
                        partial[1] = Native::AccumulateDot(partial[1], b, Native::Load<W, Lanes>(columns + N + input));
                    }

                    for (const auto sum : partial) sums = Native::Add(sums, sum);
                } else {
                    for (s00 input = 0; input < N; input += Lanes) {
                        const auto values = Act::ApplyVector(Native::Load<F, Lanes>( firstValues + input));
                        const auto column = Native::Load<W, Lanes>(columns + input);

                        sums = Native::AccumulateDot(sums, values, column);
                    }

                    for (s00 input = 0; input < N; input += Lanes) {
                        const auto values = Act::ApplyVector(Native::Load<F, Lanes>(secondValues + input));
                        const auto column = Native::Load<W, Lanes>(columns + N + input);

                        sums = Native::AccumulateDot(sums, values, column);
                    }
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
