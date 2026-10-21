//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STAGE_PAIRWISECONCAT_H
#define MANTARAY_BACKEND_STAGE_PAIRWISECONCAT_H

#include <string_view>
#include <type_traits>

#include "Format.h"
#include "../../Architecture/Signal.h"
#include "../../Architecture/Topology/Concat.h"
#include "../../Common/Integral.h"
#include "../../Common/Interval.h"
#include "../../Common/List.h"
#include "../../Common/Message.h"
#include "../../IO/Loader.h"
#include "../Kernel/Pairwise.h"

namespace MantaRay::Backend::Stage
{

    namespace Detail
    {

        struct PairwiseChoice
        {

            bool                   Valid  = false;
            Kernel::PairwiseKernel Kernel {};

        };

        template<typename A, typename ISA, s00 I>
        consteval PairwiseChoice ChoosePairwise()
        {
            using P = LayerAt<A, I>;

            if constexpr (!IsPairwise<P> || !std::is_same_v<LayerAt<A, I + 1>, Concat>)
                return {};
            else if constexpr (!Rectifier<typename P::Activation>::Known || Rectifier<typename P::Activation>::Squared)
                return {};
            else {
                constexpr Signal Input  = A::Signals[I];
                constexpr Format Output = FormatOf<A, I + 2>();

                if (FormatOf<A, I>().Encoding != Encoding::I16) return {};

                const Interval bounds = Rectifier<typename P::Activation>::Bounds(Input.Scale);
                const Interval x      = Clamp(Input.Range, bounds.Min, bounds.Max);
                const Interval clamp  = ClampOf<A, I>(bounds);

                if (!Fits<i16>(x) || P::Shift > 16) return {};

                const Kernel::PairwiseKernel kernel {
                    .Shift  = P::Shift,
                    .Low    = static_cast<i16>(clamp.Min),
                    .High   = static_cast<i16>(clamp.Max),
                    .Offset = Output.Offset
                };

                const i64 scale = P::Shift == 0 ? 1 : i64 { 1 } << (16 - P::Shift);

                const bool exact   = P::Shift == 0 && Fits<i16>(x * x                                      );
                const bool shifted = P::Shift  > 0 && Fits<i16>(x * Interval { .Min = scale, .Max = scale });

                switch (Output.Encoding) {
                    case Encoding::I16      :
                    case Encoding::U08      : return { .Valid = exact || shifted, .Kernel = kernel };
                    case Encoding::ZeroPoint: return { .Valid = P::Shift == 0   , .Kernel = kernel };
                    default                 : return {};
                }
            }
        }

    }

    template<typename A, typename ISA, s00 I>
    consteval s00 PairwiseConcatLength() { return Detail::ChoosePairwise<A, ISA, I>().Valid ? 2 : 0; }

    template<typename A, typename ISA, s00 I>
    class PairwiseConcat
    {

        constexpr static s00    H      = A::Signals[I].Size;
        constexpr static Format Output = FormatOf<A, I + 2>();

        consteval static Message Describing()
        {
            return Message {} << "PairwiseConcat<" << EncodingName(Output.Encoding) << ">";
        }

        constexpr static Message Text = Describing();

        public:
        static_assert(PairwiseConcatLength<A, ISA, I>() == 2, "PairwiseConcat does not match these layers");

        using Tensors = List<>;
        using Element = Stage::Element<Output>;

        constexpr static bool Final        = false;
        constexpr static s00  Elements     = H;
        constexpr static u08  Perspectives = 1;

        constexpr static Kernel::PairwiseKernel Plan = Detail::ChoosePairwise<A, ISA, I>().Kernel;

        struct Parameters {};

        template<typename Reader>
        static LoadResult Load(Reader&, Parameters&) { return {}; }

        constexpr static std::string_view Describe() { return { Text.data(), Text.size() }; }

        [[clang::always_inline]]
        static void Run(const Parameters&, const i16* stm, const i16* ntm, const s00, Element* output)
        {
            Kernel::Pairwise<ISA, Plan, H>(stm, ntm, output);
        }

    };

}

#endif //MANTARAY_BACKEND_STAGE_PAIRWISECONCAT_H
