//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_EMITTER_H
#define MANTARAY_BACKEND_EMITTER_H

#include <tuple>
#include <variant>

#include "Graph.h"
#include "Kernel/Activation.h"
#include "Kernel/Requantize.h"
#include "Kernel/Fused/ActivateConcatAffine.h"

namespace MantaRay::Backend
{

    namespace Detail
    {

        template<typename T, s00 N>
        struct Tensor
        {

            Array<T, N> Values {};

            constexpr static s00 Size = N;

            constexpr T operator [](const s00 index) const { return Values[index]; }

        };

        template<typename T, s00 N>
        struct TensorView
        {

            const Array<T, N>& Values;

            constexpr static s00 Size = N;

            constexpr T operator [](const s00 index) const { return Values[index]; }

        };

        template<Node::Activation A, i32 L, i32 H> struct ActivationType;

        template<i32 L, i32 H> struct ActivationType<Node::Activation::Identity, L, H> { using Type = Identity; };

        template<i32 L, i32 H>
        struct ActivationType<Node::Activation::Clipped , L, H> { using Type =        ClippedReLU<L, H>; };
        template<i32 L, i32 H>
        struct ActivationType<Node::Activation::Squared , L, H> { using Type = SquaredClippedReLU<L, H>; };

    }

    template<auto Program, bool Aligned = false>
    struct Emitter
    {

        static_assert(Program.Valid(), "Cannot emit an invalid computation graph");

        using Q = decltype(Program)::Quantization;

        using F = Q::FeatureType;
        using S = Q::    SumType;

        template<s00 Index, typename Storage, typename Input, typename Values>
        static auto Instruction(const Storage& storage, const Input& input, const u08 perspective, const Values& values)
        {
            constexpr auto node = Program.Nodes[Index];
            constexpr auto code = node.Code;

            using enum Node::Operation;

            if constexpr (code == Input) {
                if constexpr (Program.Perspectives == 0) return Detail::TensorView<F, node.Size> { input };
                else return std::monostate {};
            } else if constexpr (code == Affine && node.FeatureAffine) {
                return std::monostate {};
            } else if constexpr (code == Accumulate || code == AffineAccumulate) {
                return Detail::TensorView<F, node.Size> { input[perspective ^ node.State] };
            } else if constexpr (code == Activate) {
                using A = Detail::ActivationType<node.Function, node.Minimum, node.Maximum>::Type;
                using T = std::conditional_t<node.Unit == Node::Domain::Product, S, F>;
                using Act = Activation<A, Q>;

                const auto& source = std::get<node.Inputs[0]>(values);

                static_assert(!node.NarrowSquare || Act::SIMDCompatible, "Invalid narrow activation proof");

                constexpr s00 Lanes = Native::NativeLanes<F>;

                constexpr bool Vectorized = node.Unit == Node::Domain::Feature && Lanes > 1 &&
                    (node.Function != Node::Activation::Squared || node.NarrowSquare);

                Detail::Tensor<T, node.Size> output;

                s00 index = 0;

                if constexpr (Vectorized) {
                    for (; index + Lanes <= node.Size; index += Lanes) {
                        const auto value = Native::Load<F, Lanes>(source.Values.data() + index);

                        Native::Store(output.Values.data() + index, Act::ApplyVector(value));
                    }
                }

                for (; index < node.Size; index++) {
                    if (node.Unit == Node::Domain::Product) output.Values[index] = Act::ApplySum(source[index]);
                    else                                    output.Values[index] = Act::Apply   (source[index]);
                }

                return output;
            } else if constexpr (code == Requantize) {
                const auto& source = std::get<node.Inputs[0]>(values);

                Detail::Tensor<F, node.Size> output;

                for (s00 i = 0; i < node.Size; i++) output.Values[i] = Kernel::Requantize<Q>(source[i]);

                return output;
            } else if constexpr (code == Scale) {
                const auto& source = std::get<node.Inputs[0]>(values);
                Detail::Tensor<S, node.Size> output;

                for (s00 i = 0; i < node.Size; i++) {
                    if (Program.Nodes[node.Inputs[0]].Unit == Node::Domain::Product)
                         output.Values[i] = Kernel::FinalScale<Q>(source[i]);
                    else output.Values[i] = WrapMul(static_cast<S>(source[i]), static_cast<S>(Q::OutputScale)) / Q::QA;
                }

                return output;
            } else if constexpr (code == Add) {
                const auto&  first = std::get<node.Inputs[0]>(values);
                const auto& second = std::get<node.Inputs[1]>(values);

                Detail::Tensor<F, node.Size> output;

                for (s00 i = 0; i < node.Size; i++) output.Values[i] = WrapAdd(first[i], second[i]);

                return output;
            } else if constexpr (code == Concat) {
                Detail::Tensor<F, node.Size> output;

                s00 offset = 0;

                [&]<s00... Is>(std::index_sequence<Is...>) {
                    ([&] {
                        const auto& source = std::get<node.Inputs[Is]>(values);

                        for (s00 j = 0; j < std::remove_cvref_t<decltype(source)>::Size; j++)
                            output.Values[offset++] = source[j];
                    }(), ...);
                }(std::make_index_sequence<node.InputCount> {});

                return output;
            } else if constexpr (code == ActivateConcatAffine) {
                using A = Detail::ActivationType<node.Function, node.Minimum, node.Maximum>::Type;
                const auto& parameter = storage.template Get<node.Parameter>();

                static_assert(
                    !node.NarrowSquare || Activation<A, Q>::SIMDCompatible,
                    "Invalid narrow activation proof"
                );

                constexpr auto  First = Program.Nodes[node.Inputs[0]];
                constexpr auto Second = Program.Nodes[node.Inputs[1]];

                constexpr bool StateInputs = ( First.Code == Accumulate ||  First.Code == AffineAccumulate) &&
                                             (Second.Code == Accumulate || Second.Code == AffineAccumulate)  ;

                constexpr bool AlignedInputs = Aligned && StateInputs && First.Size * sizeof(F) % Alignment == 0 &&
                                               2 * First.Size * sizeof(typename Q::WeightType)  % Alignment == 0  ;

                return Detail::Tensor<S, node.Size> {
                    Kernel::ActivateConcatAffine<A, Q, First.Size, node.Size, node.NarrowSquare, AlignedInputs>(
                        std::get<node.Inputs[0]>(values).Values,
                        std::get<node.Inputs[1]>(values).Values,
                        parameter.Weight, parameter.Bias
                    )
                };
            } else if constexpr (code == Affine || code == ConcatAffine) {
                const auto& parameter = storage.template Get<node.Parameter>();
                Detail::Tensor<S, node.Size> output;

                for (s00 out = 0; out < node.Size; out++) {
                    S sum = parameter.Bias[out];
                    s00 offset = 0;

                    [&]<s00... Is>(std::index_sequence<Is...>) {
                        ([&] {
                            const auto& source = std::get<node.Inputs[Is]>(values);

                            for (s00 in = 0; in < std::remove_cvref_t<decltype(source)>::Size; in++) sum = WrapAdd(
                                sum,
                                WrapMul(static_cast<S>(source[in]), static_cast<S>(parameter.Weight[out][offset + in]))
                            );

                            offset += std::remove_cvref_t<decltype(source)>::Size;
                        }(), ...);
                    }(std::make_index_sequence<node.InputCount> {});

                    output.Values[out] = sum;
                }

                return output;
            }
        }

        template<s00 Index = 0, typename Storage, typename Input, typename... Values>
        static auto Schedule(
            const Storage& storage, const Input& input, const u08 perspective, std::tuple<Values...>&& values
        )
        {
            if constexpr (Index == Program.Count) {
                const auto& output = std::get<Program.Output>(values).Values;

                if constexpr (Program.Nodes[Program.Output].Size == 1) return output[0];
                else return output;
            } else {
                auto value = Instruction<Index>(storage, input, perspective, values);

                return Schedule<Index + 1>(
                    storage, input, perspective,
                    std::tuple_cat(std::move(values), std::tuple { std::move(value) })
                );
            }
        }

        template<typename Storage, typename Input>
        static auto Run(const Storage& storage, const Input& input, const u08 perspective = 0)
        {
            return Schedule(storage, input, perspective, std::tuple {});
        }

    };

}

#endif
