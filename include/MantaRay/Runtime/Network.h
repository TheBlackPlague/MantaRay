//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_NETWORK_H
#define MANTARAY_RUNTIME_NETWORK_H

#include <cassert>
#include <utility>

#include "../Backend/Compile.h"
#include "../Backend/Kernel/ArrayAdd.h"
#include "../Backend/Kernel/ArrayCopy.h"
#include "../Backend/Kernel/ArraySub.h"
#include "../Backend/Kernel/ArraySubAdd.h"
#include "../Backend/Storage/Accumulator.h"
#include "../IO/Reader.h"
#include "../IO/Writer.h"

namespace MantaRay::Runtime
{

    template<typename Architecture>
    class Network
    {

        static_assert(Backend::ValidArchitecture<Architecture>, "Incompatible or unsupported network architecture");

        using Traits = Backend::ArchitectureTraits<Architecture>;
        using Q      = Traits::Quantization;
        using F      = Q::FeatureType;
        using S      = Q::SumType;

        Backend::NetworkStorage<Architecture> Parameters_ {};

        [[clang::always_inline]]
        const auto& FeatureParameters() const requires Traits::HasAccumulator
        {
            if constexpr (Traits::PerspectiveCount == 2)
                 return std::get<0>(Parameters_.Nodes()).Inner;
            else return std::get<0>(Parameters_.Nodes())      ;
        }

        template<typename Tail, typename Input, s00... Is>
        [[clang::always_inline]]
        auto EvaluateTail(const Input& input, std::index_sequence<Is...>) const
        {
            const auto storage = std::tie(std::get<Is + 1>(Parameters_.Nodes())...);

            return Backend::LowerSequence<Q, std::tuple_element_t<Is, Tail>...>::template Run<true>(storage, input);
        }

        template<s00 I, s00 O, typename A, typename T, typename Accumulator>
        [[clang::always_inline]]
        auto EvaluateAccumulated(Layer<I, O, A, T>, const u08 perspective, const Accumulator& state) const
        {
            using View = Backend::ActivatedView<A, Q, O>;
            using Tail = Traits::Tail;

            if constexpr (Traits::PerspectiveCount == 2) {
                const auto input = std::tuple { View { state[perspective] }, View { state[perspective ^ 1] } };

                return Backend::ScaleResult<Q>(EvaluateTail<Tail>(input,
                    std::make_index_sequence<std::tuple_size_v<Tail>> {} ));
            } else {
                return Backend::ScaleResult<Q>(EvaluateTail<Tail>(View { state[0] },
                    std::make_index_sequence<std::tuple_size_v<Tail>> {} ));
            }
        }

        public:
        using ArchitectureType = Architecture;
        using QuantizationType = Q;

        using State   = Backend::Accumulator   <Architecture>;
        using Storage = Backend::NetworkStorage<Architecture>;

        Network() = default;

              Storage& Parameters()       noexcept { return Parameters_; }
        const Storage& Parameters() const noexcept { return Parameters_; }

        template<typename Stream>
        bool ReadFrom(Stream& stream) { return IO::Read(stream, Parameters_); }

        template<typename Stream>
        bool ReadLegacyV2(Stream& stream) { return IO::ReadLegacyV2(stream, Parameters_); }

        template<typename Stream>
        bool WriteTo(Stream& stream) const { return IO::Write(stream, Parameters_); }

        [[clang::always_inline]]
        void Refresh(State& state) const requires Traits::HasAccumulator
        {
            state.Bias(FeatureParameters().Bias);
        }

        [[clang::always_inline]]
        void Insert(const s00 feature, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            assert(feature < Traits::Input::Size);

            Backend::Kernel::Add(state[0], FeatureParameters().Weight[feature]);
        }

        [[clang::always_inline]]
        void Remove(const s00 feature, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            assert(feature < Traits::Input::Size);

            Backend::Kernel::Sub(state[0], FeatureParameters().Weight[feature]);
        }

        [[clang::always_inline]]
        void Move(const s00 from, const s00 to, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            assert(from < Traits::Input::Size && to < Traits::Input::Size);

            Backend::Kernel::SubAdd(
                state[0],
                FeatureParameters().Weight[from],
                FeatureParameters().Weight[ to ]
            );
        }

        [[clang::always_inline]]
        void Insert(const s00 first, const s00 second, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(first < Traits::Input::Size && second < Traits::Input::Size);

            Backend::Kernel::Add(state[0], FeatureParameters().Weight[first ]);
            Backend::Kernel::Add(state[1], FeatureParameters().Weight[second]);
        }

        [[clang::always_inline]]
        void Remove(const s00 first, const s00 second, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(first < Traits::Input::Size && second < Traits::Input::Size);

            Backend::Kernel::Sub(state[0], FeatureParameters().Weight[first ]);
            Backend::Kernel::Sub(state[1], FeatureParameters().Weight[second]);
        }

        [[clang::always_inline]]
        void MoveFeatures(const s00 f0, const s00 t0, const s00 f1, const s00 t1, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(
                f0 < Traits::Input::Size && t0 < Traits::Input::Size &&
                f1 < Traits::Input::Size && t1 < Traits::Input::Size
            );

            Backend::Kernel::SubAdd(
                state[0],
                FeatureParameters().Weight[f0],
                FeatureParameters().Weight[t0]
            );
            Backend::Kernel::SubAdd(
                state[1],
                FeatureParameters().Weight[f1],
                FeatureParameters().Weight[t1]
            );
        }

        [[clang::always_inline]]
        void Insert(const u08 piece, const u08 color, const u08 square, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && square < 64);

            Insert(
                 color      * s00{384} + piece * s00{64} +  square      ,
                (color ^ 1) * s00{384} + piece * s00{64} + (square ^ 56),
                state
            );
        }

        [[clang::always_inline]]
        void Remove(const u08 piece, const u08 color, const u08 square, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && square < 64);

            Remove(
                 color      * s00 { 384 } + piece * s00 { 64 } +  square      ,
                (color ^ 1) * s00 { 384 } + piece * s00 { 64 } + (square ^ 56),
                state
            );
        }

        [[clang::always_inline]]
        void Move(const u08 piece, const u08 color, const u08 from, const u08 to, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && from < 64 && to < 64);

            MoveFeatures(
                 color      * s00 { 384 } + piece * s00 { 64 } +   from      ,
                 color      * s00 { 384 } + piece * s00 { 64 } +    to       ,
                (color ^ 1) * s00 { 384 } + piece * s00 { 64 } +  (from ^ 56),
                (color ^ 1) * s00 { 384 } + piece * s00 { 64 } +  ( to  ^ 56),
                state
            );
        }

        [[clang::always_inline]]
        auto Evaluate(const u08 perspective, const State& state) const requires Traits::HasAccumulator
        {
            assert(perspective < Traits::PerspectiveCount);

            return EvaluateAccumulated(typename Traits::AccumulatorLayer {}, perspective, state);
        }

        [[clang::always_inline]]
        auto Evaluate(const State& state) const requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        { return Evaluate(0, state); }

        [[clang::always_inline]]
        auto Evaluate(const Array<F, Traits::Input::Size>& input) const requires (!Traits::HasAccumulator)
        {
            return [&]<typename... Stages>(std::tuple<Stages...>*) {
                const auto result = Backend::LowerSequence<Q, Stages...>::template Run<true>(
                    Parameters_.Nodes(),
                    Backend::Value<Q, Traits::Input::Size> { input }
                );

                return Backend::ScaleResult<Q>(result);
            }(static_cast<Architecture::Nodes*>(nullptr));
        }

    };

}

#endif
