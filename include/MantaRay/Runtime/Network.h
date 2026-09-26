//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_NETWORK_H
#define MANTARAY_RUNTIME_NETWORK_H

#include <cassert>
#include <memory>

#include "State.h"
#include "../Backend/Emitter.h"
#include "../Backend/GraphBuilder.h"
#include "../Backend/Kernel/ArrayAdd.h"
#include "../Backend/Kernel/ArraySub.h"
#include "../Backend/Kernel/ArraySubAdd.h"
#include "../Backend/Storage/NetworkStorage.h"
#include "../IO/Reader.h"
#include "../IO/Writer.h"

namespace MantaRay::Runtime
{

    template<typename Architecture>
    class Network
    {

        using Traits = Backend::ArchitectureTraits<Architecture>;
        using Q      = Traits::Quantization;

        static_assert(Backend::ValidArchitecture<Architecture>);

        Backend::NetworkStorage<Architecture> Parameters_ {};

        [[clang::always_inline]]
        const auto& FeatureParameters() const requires Traits::HasAccumulator
        {
            return *std::assume_aligned<Alignment>(&Parameters_.template Get<0>());
        }

        [[clang::always_inline]]
        const auto& FeatureWeight(const s00 feature) const requires Traits::HasAccumulator
        {
            const auto& row = FeatureParameters().Weight[feature];

            if (sizeof(row) % Alignment == 0) return *std::assume_aligned<Alignment>(&row);

            return row;
        }

        constexpr static s00 Feature(const u08 piece, const u08 color, const u08 square)
        {
            return s00 { color } * 384 + s00 { piece } * 64 + square;
        }

        public:
        using ArchitectureType = Architecture;
        using QuantizationType = Q;

        constexpr static auto CompiledGraph = Backend::GraphBuilder<Architecture>::Optimized;

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

            Backend::Kernel::Add(state[0], FeatureWeight(feature));
        }

        [[clang::always_inline]]
        void Remove(const s00 feature, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            assert(feature < Traits::Input::Size);

            Backend::Kernel::Sub(state[0], FeatureWeight(feature));
        }

        [[clang::always_inline]]
        void Move(const s00 from, const s00 to, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            assert(from < Traits::Input::Size && to < Traits::Input::Size);

            Backend::Kernel::SubAdd(state[0], FeatureWeight(from), FeatureWeight(to));
        }

        [[clang::always_inline]]
        void Insert(const s00 first, const s00 second, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(first < Traits::Input::Size && second < Traits::Input::Size);

            Backend::Kernel::Add(state[0], FeatureWeight(first ));
            Backend::Kernel::Add(state[1], FeatureWeight(second));
        }

        [[clang::always_inline]]
        void Remove(const s00 first, const s00 second, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(first < Traits::Input::Size && second < Traits::Input::Size);

            Backend::Kernel::Sub(state[0], FeatureWeight(first ));
            Backend::Kernel::Sub(state[1], FeatureWeight(second));
        }

        [[clang::always_inline]]
        void MoveFeatures(const s00 f0, const s00 t0, const s00 f1, const s00 t1, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2)
        {
            assert(f0 < Traits::Input::Size && t0 < Traits::Input::Size);
            assert(f1 < Traits::Input::Size && t1 < Traits::Input::Size);

            Backend::Kernel::SubAdd(state[0], FeatureWeight(f0), FeatureWeight(t0));
            Backend::Kernel::SubAdd(state[1], FeatureWeight(f1), FeatureWeight(t1));
        }

        [[clang::always_inline]]
        void Insert(const u08 piece, const u08 color, const u08 square, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && square < 64);

            Insert(Feature(piece, color, square), Feature(piece, color ^ 1, square ^ 56), state);
        }

        [[clang::always_inline]]
        void Remove(const u08 piece, const u08 color, const u08 square, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && square < 64);

            Remove(Feature(piece, color, square), Feature(piece, color ^ 1, square ^ 56), state);
        }

        [[clang::always_inline]]
        void Move(const u08 piece, const u08 color, const u08 from, const u08 to, State& state) const
        requires (Traits::HasAccumulator && Traits::PerspectiveCount == 2 && Traits::Input::Size == 768)
        {
            assert(piece < 6 && color < 2 && from < 64 && to < 64);

            MoveFeatures(
                Feature(piece, color    , from     ), Feature(piece, color    , to     ),
                Feature(piece, color ^ 1, from ^ 56), Feature(piece, color ^ 1, to ^ 56),
                state
            );
        }

        [[clang::always_inline]]
        auto Evaluate(const u08 perspective, const State& state) const requires Traits::HasAccumulator
        {
            assert(perspective < Traits::PerspectiveCount);

            return Backend::Emitter<CompiledGraph, true>::Run(Parameters_, state, perspective);
        }

        [[clang::always_inline]]
        auto Evaluate(const State& state) const requires (Traits::HasAccumulator && Traits::PerspectiveCount == 1)
        {
            return Evaluate(0, state);
        }

        [[clang::always_inline]]
        auto Evaluate(const Array<typename Q::FeatureType, Traits::Input::Size>& input) const
        requires (!Traits::HasAccumulator)
        {
            return Backend::Emitter<CompiledGraph, true>::Run(Parameters_, input);
        }

    };

}

#endif
