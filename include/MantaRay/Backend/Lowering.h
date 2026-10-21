//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_LOWERING_H
#define MANTARAY_BACKEND_LOWERING_H

#include <array>
#include <string_view>
#include <type_traits>
#include <utility>

#include "../Architecture/Signal.h"
#include "../Common/Alignment.h"
#include "../Common/Integral.h"
#include "../Common/List.h"
#include "../IO/Loader.h"
#include "Kernel/Update.h"
#include "Stage/DenseAffine.h"
#include "Stage/Format.h"
#include "Stage/PairwiseConcat.h"
#include "Stage/PerspectiveDot.h"
#include "Stage/Single.h"

namespace MantaRay::Backend
{

    namespace Detail
    {

        template<typename A, typename ISA, s00 I, typename... S>
        consteval auto Lower(List<S...>)
        {
            if constexpr (I == A::LayerCount)
                return List<S...> {};
            else if constexpr (constexpr s00 N = Stage::PerspectiveDotLength<A, ISA, I>(); N > 0)
                return Lower<A, ISA, I + N>(List<S..., Stage::PerspectiveDot<A, ISA, I, N>> {});
            else if constexpr (constexpr s00 N = Stage::PairwiseConcatLength<A, ISA, I>(); N > 0) {
                static_assert(
                    Stage::FormatOf<A, I + N>().Encoding != Stage::Encoding::U08 ||
                    Stage::DenseAffineLength<A, ISA, I + N>() > 0                 ,
                    "packed u8 pairwise products must feed a DenseAffine, which undoes the pack order"
                );

                return Lower<A, ISA, I + N>(List<S..., Stage::PairwiseConcat<A, ISA, I   >> {});
            } else if constexpr (constexpr s00 N = Stage::DenseAffineLength<A, ISA, I>(); N > 0)
                return Lower<A, ISA, I + N>(List<S..., Stage::DenseAffine   <A, ISA, I, N>> {});
            else
                return Lower<A, ISA, I + 1>(List<S..., Stage::Single        <A, ISA, I   >> {});
        }

        template<typename S>
        constexpr inline bool IsPerspectiveDot = false;

        template<typename A, typename ISA, s00 I, s00 N>
        constexpr inline bool IsPerspectiveDot<Stage::PerspectiveDot<A, ISA, I, N>> = true;

        template<s00 I, typename P>
        struct Slot : P {};

        template<typename Sequence, typename... S>
        struct Slots;

        template<s00... I, typename... S>
        struct Slots<std::index_sequence<I...>, S...> : Slot<I, typename S::Parameters>... {};

        template<typename... S>
        consteval auto Storage(List<S...>) -> Slots<std::index_sequence_for<S...>, S...>;

        constexpr inline std::string_view Separator = " -> ";

        template<typename... S>
        consteval s00 Length(List<S...>)
        {
            return (S::Describe().size() + ... + 0) + Separator.size() * (sizeof...(S) - 1);
        }

        template<s00 N, typename... S>
        consteval std::array<char, N> Join(List<S...>)
        {
            std::array<char, N> text {};

            s00 length = 0;

            const auto append = [&](const std::string_view part) { for (const char c : part) text[length++] = c; };

            ([&] {
                if (length > 0) append(Separator);

                append(S::Describe());
            }(), ...);

            return text;
        }

        template<typename S>
        constexpr inline auto Description = Join<Length(S {})>(S {});

        template<typename P, s00 I, typename T>
        [[clang::always_inline]]
        i32 Run(const typename P::Parameters& parameters, const T* stm, const T* ntm, const s00 bucket)
        {
            using S = P::template StageAt<I>;

            if constexpr (S::Final) return S::Run(parameters.template Get<I>(), stm, ntm, bucket);
            else {
                alignas(Alignment) std::array<typename S::Element, S::Elements> output;

                S::Run(parameters.template Get<I>(), stm, ntm, bucket, output.data());

                const typename S::Element* values = output.data();
                const typename S::Element* second = S::Perspectives == 2 ? values + S::Elements / 2 : nullptr;

                return Run<P, I + 1>(parameters, values, second, bucket);
            }
        }

        template<typename P, s00 I, typename Reader>
        LoadResult Load(Reader& reader, typename P::Parameters& parameters)
        {
            if constexpr (I == P::Stages::Size) return {};
            else {
                using S = P::template StageAt<I>;

                if (const LoadResult result = S::Load(reader, parameters.template Get<I>()); !result) return result;

                return Load<P, I + 1>(reader, parameters);
            }
        }

    }

    template<typename A, typename ISA>
    struct Program
    {

        using Stages = decltype(Detail::Lower<A, ISA, 0>(List<> {}));

        using Accumulator = std::conditional_t<AccumulatorBytes(A::Signals[0].Range) == 2, i16, i32>;

        template<s00 I>
        using StageAt = ItemAt<Stages, I>;

        constexpr static bool Fusable = Detail::IsPerspectiveDot<StageAt<0>>;

        struct Parameters : decltype(Detail::Storage(Stages {}))
        {

            template<s00 I>
            [[clang::always_inline]]
                  StageAt<I>::Parameters& Get()
            { return static_cast<Detail::Slot<I, typename StageAt<I>::Parameters>&>(*this); }
            template<s00 I>
            [[clang::always_inline]]
            const StageAt<I>::Parameters& Get() const
            { return static_cast<const Detail::Slot<I, typename StageAt<I>::Parameters>&>(*this); }

        };

        constexpr static std::string_view Describe()
        {
            return { Detail::Description<Stages>.data(), Detail::Description<Stages>.size() };
        }

        template<typename Reader>
        static LoadResult Load(Reader& reader, Parameters& parameters)
        {
            return Detail::Load<Program, 0>(reader, parameters);
        }

        [[clang::always_inline]]
        static i32 Forward(
            const Parameters&  parameters,
            const Accumulator* stm       ,
            const Accumulator* ntm       ,
            const s00          bucket
        )
        {
            return Detail::Run<Program, 0>(parameters, stm, ntm, bucket);
        }

        template<s00 Subs, s00 Adds>
        [[clang::always_inline]]
        static i32 ForwardFused(
            const Parameters&                                  parameters,
            const Kernel::Transition<Accumulator, Subs, Adds>& transition,
            const s00                                          bucket
        ) requires (Fusable)
        {
            return StageAt<0>::template RunFused<Subs, Adds>(parameters.template Get<0>(), transition, bucket);
        }

    };

}

#endif //MANTARAY_BACKEND_LOWERING_H
