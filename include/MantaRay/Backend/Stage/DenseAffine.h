//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STAGE_DENSEAFFINE_H
#define MANTARAY_BACKEND_STAGE_DENSEAFFINE_H

#include <algorithm>
#include <cassert>
#include <span>
#include <string_view>
#include <type_traits>

#include "Format.h"
#include "PairwiseConcat.h"
#include "../Facts.h"
#include "../../Common.h"
#include "../../Architecture/Signal.h"
#include "../../IO/Loader.h"
#include "../Kernel/Dense.h"
#include "../Kernel/Pack.h"

namespace MantaRay::Backend::Stage
{

    struct DensePlan
    {

        bool   Valid        = false;
        Format Input        {};
        s00    Accumulators = 1;
        bool   WideReduce   = false;

    };

    namespace Detail
    {

        template<typename A, s00 I>
        consteval s00 DenseSpan()
        {
            constexpr s00 S = I + 1 + IsSelect   <LayerAt<A, I + 1>>;
            constexpr s00 T = S     + Elementwise<LayerAt<A, S    >>;
            constexpr s00 U = T     + IsOutput   <LayerAt<A, T    >>;

            return U - I;
        }

        template<typename A, typename ISA, s00 I>
        consteval DensePlan ChooseDense()
        {
            using D = LayerAt<A, I>;

            if constexpr (!IsAffine<D>) return {};
            else {
                constexpr Format Input = FormatOf<A, I>();

                if (D::IsFloat) return { .Valid = true, .Input = Input };

                if (Input.Encoding != Encoding::I16       &&
                    Input.Encoding != Encoding::ZeroPoint &&
                    Input.Encoding != Encoding::U08       ) return {};

                const Interval range   = A::Signals[I].Range;
                const Interval x       = { .Min = range.Min - Input.Offset, .Max = range.Max - Input.Offset };
                const Interval weights = D::WeightSpec::Bounds;

                if (Input.Encoding == Encoding::U08 ? !MaddubsExact(x, weights) : !MaddExact(x, weights)) return {};

                constexpr s00 Width = sizeof(Element<Input>);

                if (D::InputSize * Width % VectorBytes != 0) return {};

                const Interval product = x * weights;
                const s00      vectors = D::InputSize * Width / ISA::Bytes;
                const DotPlan  plan    = PlanDot<ISA>(product, D::InputSize);
                const s00      chains  = std::min(plan.Accumulators, vectors);
                const s00      partial = sizeof(i32) / Width * ((vectors + chains - 1) / chains);

                if (plan.WideReduce && !Fits<i32>(Repeat(product, partial))) return {};

                return { .Valid = true, .Input = Input, .Accumulators = chains, .WideReduce = plan.WideReduce };
            }
        }

        template<typename A, typename ISA, s00 I>
        consteval bool PackedInput()
        {
            if (I < 2) return false;

            return FormatOf<A, I>().Encoding == Encoding::U08 && PairwiseConcatLength<A, ISA, I - 2>() == 2;
        }

    }

    template<typename A, typename ISA, s00 I>
    consteval s00 DenseAffineLength() { return Detail::ChooseDense<A, ISA, I>().Valid ? Detail::DenseSpan<A, I>() : 0; }

    template<typename A, typename ISA, s00 I, s00 N>
    class DenseAffine
    {

        using D = LayerAt<A, I>;

        using W = D::WeightSpec;
        using B = D::  BiasSpec;

        constexpr static bool Selects   = IsSelect<LayerAt<A, I + 1>>;
        constexpr static s00  Activated = I + 1 + Selects;
        constexpr static bool Activates = Activated < I + N && Elementwise<LayerAt<A, Activated>>;
        constexpr static s00  Exited    = Activated + Activates;
        constexpr static bool Exits     = Exited < I + N;

        using Activation = LayerAt<A, Activated>;
        using Exit       = LayerAt<A, Exited   >;

        constexpr static s00 Inputs  = D:: InputSize;
        constexpr static s00 Outputs = D::OutputSize;

        constexpr static s00 Buckets = A::Signals[I + 1].Buckets;

        constexpr static s00 Rows = Outputs / Buckets;

        constexpr static DensePlan Choice = Detail::ChooseDense<A, ISA, I>();

        constexpr static Format Source = Choice.Input;
        constexpr static Format Target = FormatOf<A, I + N>();

        constexpr static bool Real    = D::IsFloat || D::IsDequantize;
        constexpr static bool Shifted = Source.Encoding == Encoding::ZeroPoint;
        constexpr static bool Packed  = Detail::PackedInput<A, ISA, I>();

        using Input = Element<Source>;

        using Weight = std::conditional_t<D::IsFloat, f32, std::conditional_t<std::is_same_v<Input, u08>, i08, i16>>;
        using Bias   = std::conditional_t<Real, f32, i32>;

        constexpr static f32 Factor = D::DequantizeFactor(A::Signals[I]);

        [[clang::always_inline]]
        static auto Evaluate(
            const Array<Input , Inputs>& input  ,
            const Array<Weight, Inputs>& weights,
            const Bias                   bias   ,
            const i64                    offset
        )
        {
            #pragma clang fp contract(off)

            if constexpr (D::IsFloat) return Kernel::DenseDotF<ISA, Inputs>(input, weights) + bias;
            else {
                const i64 sum = Kernel::DenseDot<ISA, Inputs, Choice.Accumulators, Choice.WideReduce>(input, weights) +
                                offset;

                if constexpr (D::IsDequantize) return static_cast<f32>(sum) * Factor + bias;
                else                           return D::Schedule::Apply(sum) + i64 { bias };
            }
        }

        template<typename V>
        [[clang::always_inline]]
        static V Activate(const V value)
        {
            if constexpr (!Activates) return value;
            else {
                if (Real) return Activation::ApplyFloat(value);

                return Activation::Apply(value, A::Signals[Activated].Scale);
            }
        }

        consteval static Message Describing()
        {
            Message text {};

            text << "DenseAffine<" << EncodingName(Source.Encoding);

            if (D::IsDequantize) text << ",Dequantize";
            if (Selects        ) text << ",Select=" << static_cast<i64>(Buckets);

            if (Activates) text << "," << ShortName<Activation>();

            if (Exits            ) text << ",Output";
            if (Choice.WideReduce) text << ",i64"   ;

            return text << ">";
        }

        constexpr static Message Text = Describing();

        public:
        static_assert(DenseAffineLength<A, ISA, I>() == N, "DenseAffine does not match these layers");

        using Tensors = List<W, B>;
        using Element = Stage::Element<Target>;
        using Buffer  = Array<Element, 1, A::Signals[I + N].Size>;

        constexpr static bool Final        = Exits;
        constexpr static u08  Perspectives = 1;

        constexpr static DensePlan Plan = Choice;

        struct Parameters
        {

            alignas(Alignment) Array<Weight, Outputs, Inputs> Weights;
            alignas(Alignment) Array<Bias  , Outputs        > Biases ;

            NO_UNIQUE_ADDRESS std::conditional_t<Shifted, Array<i64, Outputs>, Absent<0>> Offsets;

        };

        static void Pack(
            const std::span<const typename W::Type> weights   ,
            const std::span<const typename B::Type> biases    ,
                  Parameters&                       parameters
        )
        {
            assert(weights.size() == Outputs * Inputs && biases.size() == Outputs);

            for (s00 o = 0; o < Outputs; o++) {
                i64 total = 0;

                for (s00 i = 0; i < Inputs; i++) {
                    const s00  input = Kernel::LogicalIndex<ISA, Packed>(i);
                    const auto value = weights[BulletIndex<W>(Inputs, Outputs, input, o)];

                    parameters.Weights[o][i] = static_cast<Weight>(value);

                    if (!D::IsFloat) total += value;
                }

                parameters.Biases[o] = static_cast<Bias>(biases[o]);

                if constexpr (Shifted) parameters.Offsets[o] = Source.Offset * total;
            }
        }

        template<typename Reader>
        static LoadResult Load(Reader& reader, Parameters& parameters)
        {
            return LoadAffine<D, DenseAffine>(reader, parameters);
        }

        constexpr static std::string_view Describe() { return { Text.data(), Text.size() }; }

        [[clang::always_inline]]
        static void Run(
            const Parameters&           parameters,
            const Array<Input, Inputs>& input     ,
            const Array<Input, Inputs>&           ,
            const s00                   bucket    ,
                  Buffer&               output
        ) requires (!Final)
        {
            assert(bucket < Buckets || Buckets == 1);

            const s00 first = Buckets == 1 ? 0 : bucket * Rows;

            for (s00 m = 0; m < Rows; m++) {
                const s00 row = first + m;

                i64 offset = 0;

                if constexpr (Shifted) offset = parameters.Offsets[row];

                const Array<Weight, Inputs>& weights = parameters.Weights[row];

                output[0][m] = Write<Target>(Activate(Evaluate(input, weights, parameters.Biases[row], offset)));
            }
        }

        [[clang::always_inline]]
        static i32 Run(
            const Parameters&           parameters,
            const Array<Input, Inputs>& input     ,
            const Array<Input, Inputs>&           ,
            const s00                   bucket
        ) requires (Final)
        {
            assert(bucket < Buckets || Buckets == 1);

            const s00 row = Buckets == 1 ? 0 : bucket;

            i64 offset = 0;

            if constexpr (Shifted) offset = parameters.Offsets[row];

            const Array<Weight, Inputs>& weights = parameters.Weights[row];

            const auto value = Activate(Evaluate(input, weights, parameters.Biases[row], offset));

            if constexpr (Real) return Exit::ApplyFloat(value);
            else                return Exit::template Apply<A::Signals[Exited]>(value);
        }

    };

}

#endif //MANTARAY_BACKEND_STAGE_DENSEAFFINE_H
