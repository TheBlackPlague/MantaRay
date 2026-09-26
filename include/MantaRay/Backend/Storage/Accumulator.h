//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STORAGE_ACCUMULATOR_H
#define MANTARAY_BACKEND_STORAGE_ACCUMULATOR_H

#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <utility>

#include "../Traits.h"
#include "../../Common/Alignment.h"
#include "../../Common/Container.h"
#include "../Kernel/ArrayCopy.h"

namespace MantaRay::Backend
{

    template<typename Architecture>
    struct Accumulator
    {

        using Traits      = ArchitectureTraits<Architecture>;
        using FeatureType = Traits::Quantization::FeatureType;
        using Layer       = Traits::AccumulatorLayer;

        static_assert(Traits::HasAccumulator);

        constexpr static s00 HiddenSize       = Layer::OutputSize;
        constexpr static s00 PerspectiveCount = Traits::PerspectiveCount;

        using Row = Array<FeatureType, HiddenSize>;

        ALIGN std::array<Row, PerspectiveCount> Values;

        Accumulator() { Zero(); }

        [[clang::always_inline]]
        Accumulator(const Accumulator& source)
        {
            Kernel::Copy(
                *std::assume_aligned<Alignment>(&source.Values),
                *std::assume_aligned<Alignment>(&Values)
            );
        }

        [[clang::always_inline]]
        Accumulator& operator =(const Accumulator& source)
        {
            Kernel::Copy(
                *std::assume_aligned<Alignment>(&source.Values),
                *std::assume_aligned<Alignment>(&Values)
            );

            return *this;
        }

        void Zero() { std::memset(Values.data(), 0, sizeof(Values)); }

        [[clang::always_inline]]
        void Bias(const Row& value)
        {
            [&]<s00... Side>(std::index_sequence<Side...>) {
                (Kernel::Copy(value, (*this)[Side]), ...);
            }(std::make_index_sequence<PerspectiveCount> {});
        }

        Row& operator [](const s00 side)
        {
            assert(side < PerspectiveCount);

            if (sizeof(Row) % Alignment == 0) return *std::assume_aligned<Alignment>(&Values[side]);

            return Values[side];
        }

        const Row& operator [](const s00 side) const
        {
            assert(side < PerspectiveCount);

            if (sizeof(Row) % Alignment == 0) return *std::assume_aligned<Alignment>(&Values[side]);

            return Values[side];
        }

    };

}

#endif
