//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STAGE_PERSPECTIVEDOT_H
#define MANTARAY_BACKEND_STAGE_PERSPECTIVEDOT_H

#include <algorithm>
#include <cassert>
#include <span>
#include <string_view>
#include <type_traits>

#include "Format.h"
#include "../Facts.h"
#include "../../Common.h"
#include "../../Architecture/Signal.h"
#include "../../Architecture/Topology/Concat.h"
#include "../../IO/Loader.h"
#include "../Kernel/Dot.h"
#include "../Kernel/Pack.h"
#include "../Kernel/Update.h"

namespace MantaRay::Backend::Stage
{

    namespace Detail
    {

        struct DotChoice
        {

            bool              Valid  = false;
            s00               Length = 0;
            Kernel::DotKernel Kernel {};

        };

        template<typename A, typename ISA, s00 I>
        consteval DotChoice ChooseDot()
        {
            constexpr s00 Length = IsSelect<LayerAt<A, I + 3>> ? 5 : 4;

            using R = LayerAt<A, I             >;
            using D = LayerAt<A, I + 2         >;
            using O = LayerAt<A, I + Length - 1>;

            if constexpr (!Rectifier<R>::Known || !std::is_same_v<LayerAt<A, I + 1>, Concat> || !IsOutput<O>)
                return {};
            else if constexpr (!IsAffine<D>)
                return {};
            else {
                if (D::IsFloat || D::IsDequantize) return {};

                constexpr Signal Input = A::Signals[I];

                if (Input.Perspectives != 2 || FormatOf<A, I>().Encoding != Encoding::I16) return {};

                const Interval weights = D::WeightSpec::Bounds;
                const Interval bounds  = Rectifier<R>::Bounds(Input.Scale);
                const Interval x       = Clamp(Input.Range, bounds.Min, bounds.Max);
                const Interval clamp   = ClampOf<A, I>(bounds);

                if (!Fits<i16>(weights) || !Fits<i16>(x)) return {};

                Kernel::DotKernel kernel {
                    .Low  = static_cast<i16>(clamp.Min),
                    .High = static_cast<i16>(clamp.Max)
                };

                Interval product = x * weights;

                if (!Rectifier<R>::Squared) {
                    if (!MaddExact(x, weights)) return {};

                    const bool packed = ISA::VNNI && PackusIsCReLU(bounds) && FitsI8(weights) &&
                                        Input.Size % ISA::Bytes == 0;

                    kernel.Mode = packed ? Kernel::DotMode::PackusDot : Kernel::DotMode::Linear;
                } else {
                    product = Square(x) * weights;

                    const bool pairs = Fits<i32>(Repeat(product, 2));

                    if      (pairs && LizardExact(x, weights)) kernel.Mode = Kernel::DotMode::Lizard     ;
                    else if (pairs && SquareFitsI16(x)       ) kernel.Mode = Kernel::DotMode::SquareFirst;
                    else if (Fits<i32>(product)              ) kernel.Mode = Kernel::DotMode::Widened    ;
                    else return {};
                }

                const bool    packed  = kernel.Mode == Kernel::DotMode::PackusDot;
                const s00     width   = packed ? ISA::Bytes : ISA::Bytes / sizeof(i16);
                const s00     terms   = packed ? 4 : 2;
                const s00     vectors = 2 * Input.Size / width;
                const DotPlan plan    = PlanDot<ISA>(product, 2 * Input.Size);

                kernel.Accumulators = std::min(plan.Accumulators, vectors);
                kernel.WideReduce   = plan.WideReduce;

                const s00 partial = terms * ((vectors + kernel.Accumulators - 1) / kernel.Accumulators);

                if (plan.WideReduce && !Fits<i32>(Repeat(product, partial))) return {};

                return { .Valid = true, .Length = Length, .Kernel = kernel };
            }
        }

        consteval std::string_view ModeName(const Kernel::DotMode mode)
        {
            constexpr Array<std::string_view, 5> Names { "Linear", "Lizard", "SquareFirst", "Widened", "PackusDot" };

            return Names[static_cast<s00>(mode)];
        }

    }

    template<typename A, typename ISA, s00 I>
    consteval s00 PerspectiveDotLength()
    {
        constexpr Detail::DotChoice Choice = Detail::ChooseDot<A, ISA, I>();

        return Choice.Valid ? Choice.Length : 0;
    }

    template<typename A, typename ISA, s00 I, s00 N>
    class PerspectiveDot
    {

        using R = LayerAt<A, I        >;
        using D = LayerAt<A, I + 2    >;
        using O = LayerAt<A, I + N - 1>;

        using W = D::WeightSpec;
        using B = D::  BiasSpec;

        constexpr static s00 H = A::Signals[I].Size;

        constexpr static s00 Inputs  = D:: InputSize;
        constexpr static s00 Buckets = D::OutputSize;

        constexpr static bool Packed = Detail::ChooseDot<A, ISA, I>().Kernel.Mode == Kernel::DotMode::PackusDot;

        using Weight = std::conditional_t<Packed, i08, i16>;

        consteval static Message Describing()
        {
            Message text {};

            text << "PerspectiveDot<" << Rectifier<R>::Name;

            if (Rectifier<R>::Squared || Packed) text << "-" << Detail::ModeName(Plan.Mode);

            text << ",K=" << static_cast<i64>(Plan.Accumulators) << "," << (Plan.WideReduce ? "i64" : "i32");

            if (Buckets > 1) text << ",Select=" << static_cast<i64>(Buckets);

            return text << ">";
        }

        constexpr static Message Text = Describing();

        [[clang::always_inline]]
        static i32 Finish(const i64 sum, const i32 bias)
        {
            return O::template Apply<A::Signals[I + N - 1]>(D::Schedule::Apply(sum) + bias);
        }

        public:
        static_assert(PerspectiveDotLength<A, ISA, I>() == N, "PerspectiveDot does not match these layers");

        using Tensors = List<W, B>;

        constexpr static bool Final = true;

        constexpr static Kernel::DotKernel Plan = Detail::ChooseDot<A, ISA, I>().Kernel;

        struct Parameters
        {

            alignas(Alignment) Array<Weight, Buckets, 2, H> Weights;
            alignas(Alignment) Array<i32   , Buckets      > Biases ;

        };

        static void Pack(
            const std::span<const typename W::Type> weights   ,
            const std::span<const typename B::Type> biases    ,
                  Parameters&                       parameters
        )
        {
            assert(weights.size() == Buckets * Inputs && biases.size() == Buckets);

            for (s00 o = 0; o < Buckets; o++) {
                for (s00 i = 0; i < Inputs; i++) {
                    const s00  input = Kernel::LogicalIndex<ISA, Packed>(i);
                    const auto value = weights[BulletIndex<W>(Inputs, Buckets, input, o)];

                    parameters.Weights[o][i / H][i % H] = static_cast<Weight>(value);
                }

                parameters.Biases[o] = static_cast<i32>(biases[o]);
            }
        }

        template<typename Reader>
        static LoadResult Load(Reader& reader, Parameters& parameters)
        {
            return LoadAffine<D, PerspectiveDot>(reader, parameters);
        }

        constexpr static std::string_view Describe() { return { Text.data(), Text.size() }; }

        [[clang::always_inline]]
        static i32 Run(
            const Parameters&    parameters,
            const Array<i16, H>& stm       ,
            const Array<i16, H>& ntm       ,
            const s00            bucket
        )
        {
            assert(bucket < Buckets || Buckets == 1);

            const s00 row = Buckets == 1 ? 0 : bucket;

            const i64 sum = Kernel::PerspectiveDot<ISA, Plan, H>(stm, ntm, parameters.Weights[row]);

            return Finish(sum, parameters.Biases[row]);
        }

        template<s00 Subs, s00 Adds>
        [[clang::always_inline]]
        static i32 RunFused(
            const Parameters&                             parameters,
            const Kernel::Transition<i16, H, Subs, Adds>& transition,
            const s00                                     bucket
        )
        {
            assert(bucket < Buckets || Buckets == 1);

            const s00 row = Buckets == 1 ? 0 : bucket;

            const i64 sum = Kernel::PerspectiveDot<ISA, Plan, H>(transition, parameters.Weights[row]);

            return Finish(sum, parameters.Biases[row]);
        }

    };

}

#endif //MANTARAY_BACKEND_STAGE_PERSPECTIVEDOT_H
