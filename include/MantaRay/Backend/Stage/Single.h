//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STAGE_SINGLE_H
#define MANTARAY_BACKEND_STAGE_SINGLE_H

#include <array>
#include <cassert>
#include <span>
#include <string_view>
#include <type_traits>

#include "Format.h"
#include "../../Architecture/Signal.h"
#include "../../Architecture/Topology/Concat.h"
#include "../../Common/Alignment.h"
#include "../../Common/Integral.h"
#include "../../Common/Interval.h"
#include "../../Common/List.h"
#include "../../Common/Message.h"
#include "../../IO/Loader.h"
#include "../Kernel/Dense.h"

namespace MantaRay::Backend::Stage
{

    struct SinglePlan
    {

        Format Input ;
        Format Output;

    };

    namespace Detail
    {

        template<typename L>
        struct SingleTensors
        {

            using Tensors = List<>;

            struct Parameters {};

            template<typename Reader>
            static LoadResult Load(Reader&, Parameters&) { return {}; }

        };

        template<s00 In, s00 Out, typename W, typename B, typename S>
        struct SingleTensors<Affine<In, Out, W, B, S>>
        {

            using Tensors = List<W, B>;

            struct Parameters
            {

                alignas(Alignment) std::array<typename W::Type, Out * In> Weights;
                alignas(Alignment) std::array<typename B::Type, Out     > Biases ;

            };

            static void Pack(
                const std::span<const typename W::Type> weights   ,
                const std::span<const typename B::Type> biases    ,
                      Parameters&                       parameters
            )
            {
                assert(weights.size() == Out * In && biases.size() == Out);

                for (s00 o = 0; o < Out; o++) {
                    for (s00 i = 0; i < In; i++)
                        parameters.Weights[o * In + i] = weights[BulletIndex<W>(In, Out, i, o)];

                    parameters.Biases[o] = biases[o];
                }
            }

            template<typename Reader>
            static LoadResult Load(Reader& reader, Parameters& parameters)
            {
                return LoadAffine<Affine<In, Out, W, B, S>, SingleTensors>(reader, parameters);
            }

        };

    }

    template<typename A, typename ISA, s00 I>
    class Single : public Detail::SingleTensors<LayerAt<A, I>>
    {

        using L      = LayerAt<A, I>;
        using Layout = Detail::SingleTensors<L>;

        constexpr static Signal In  = A::Signals[I    ];
        constexpr static Signal Out = A::Signals[I + 1];

        constexpr static Format Source = FormatOf<A, I    >();
        constexpr static Format Target = FormatOf<A, I + 1>();

        using Input = Stage::Element<Source>;

        constexpr static Message Text = Message {} << "Single<" << L::Name << ">";

        [[clang::always_inline]]
        static auto Activate(const Input value)
        {
            if constexpr (In.Domain == Domain::Float) return L::ApplyFloat(Read<Source>(value));
            else                                      return L::Apply     (Read<Source>(value), In.Scale);
        }

        public:
        using Tensors    = Layout::Tensors   ;
        using Parameters = Layout::Parameters;

        using Element = Element<Target>;

        constexpr static bool Final        = IsOutput<L>;
        constexpr static s00  Elements     = Out.Size * Out.Perspectives;
        constexpr static u08  Perspectives = Out.Perspectives;

        constexpr static SinglePlan Plan { .Input = Source, .Output = Target };

        constexpr static std::string_view Describe() { return { Text.data(), Text.size() }; }

        [[clang::always_inline]]
        static void Run(
            const Parameters& parameters,
            const Input*      stm       ,
            const Input*      ntm       ,
            const s00         bucket    ,
                  Element*    output
        ) requires (!Final)
        {
            #pragma clang fp contract(off)

            if constexpr (Elementwise<L>) {
                for (s00 p = 0; p < In.Perspectives; p++)
                for (s00 j = 0; j < In.Size        ; j++)
                    output[p * In.Size + j] = Write<Target>(Activate((p ? ntm : stm)[j]));
            } else if constexpr (IsPairwise<L>) {
                constexpr s00 Half = In.Size / 2;

                for (s00 p = 0; p < 2   ; p++)
                for (s00 j = 0; j < Half; j++) {
                    const Input* source = p ? ntm : stm;

                    const i64 a = L::Activation::Apply(Read<Source>(source[       j]), In.Scale);
                    const i64 b = L::Activation::Apply(Read<Source>(source[Half + j]), In.Scale);

                    output[p * Half + j] = Write<Target>(a * b >> L::Shift);
                }
            } else if constexpr (std::is_same_v<L, Concat>) {
                for (s00 p = 0; p < 2      ; p++)
                for (s00 j = 0; j < In.Size; j++)
                    output[p * In.Size + j] = Write<Target>(Read<Source>((p ? ntm : stm)[j]));
            } else if constexpr (IsSelect<L>) {
                assert(bucket < L::Buckets::Count);

                for (s00 j = 0; j < Out.Size; j++) output[j] = Write<Target>(Read<Source>(stm[bucket * Out.Size + j]));
            } else if constexpr (IsAffine<L>) {
                constexpr f32 Factor = L::DequantizeFactor(In);

                for (s00 o = 0; o < L::OutputSize; o++) {
                    const auto* weights = parameters.Weights.data() + o * L::InputSize;

                    if constexpr (L::IsFloat) {
                        output[o] = Kernel::DenseDotF<ISA, L::InputSize>(stm, weights) + parameters.Biases[o];
                    } else {
                        i64 sum = 0;

                        for (s00 i = 0; i < L::InputSize; i++) sum += Read<Source>(stm[i]) * i64 { weights[i] };

                        if constexpr (L::IsDequantize)
                            output[o] = static_cast<f32>(sum) * Factor + parameters.Biases[o];
                        else
                            output[o] = Write<Target>(L::Schedule::Apply(sum) + i64 { parameters.Biases[o] });
                    }
                }
            } else static_assert(false, "Single has no reference implementation for this layer");
        }

        [[clang::always_inline]]
        static i32 Run(const Parameters&, const Input* stm, const Input*, const s00) requires (Final)
        {
            if constexpr (In.Domain == Domain::Float) return L::ApplyFloat(stm[0]);
            else                                      return L::template Apply<In>(Read<Source>(stm[0]));
        }

    };

}

#endif //MANTARAY_BACKEND_STAGE_SINGLE_H
