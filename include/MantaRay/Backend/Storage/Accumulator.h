//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BACKEND_ACCUMULATOR_H
#define MANTARAY_BACKEND_ACCUMULATOR_H

#include <array>
#include <cassert>
#include <cstring>

#include "../Traits.h"
#include "../Kernel/ArrayCopy.h"

namespace MantaRay::Backend
{

    template<typename Architecture>
    struct Accumulator
    {

        using Traits = ArchitectureTraits<Architecture>;

        static_assert(Traits::HasAccumulator, "This architecture has no incremental feature transformer.");

        using FeatureType = Traits::Quantization::FeatureType;
        using Layer       = Traits::AccumulatorLayer         ;

        static constexpr s00 HiddenSize = Layer::OutputSize;

        static constexpr s00 PerspectiveCount = Traits::PerspectiveCount;

        using Row = std::array<FeatureType, HiddenSize>;

        ALIGN std::array<Row, PerspectiveCount> Values;

        Accumulator() { Zero(); }

        [[clang::always_inline]]
        Accumulator(const Accumulator& other) { Kernel::Copy(other.Values, Values); }

        [[clang::always_inline]]
        Accumulator& operator=(const Accumulator& other)
        {
            Kernel::Copy(other.Values, Values);

            return *this;
        }

        void Zero() { std::memset(Values.data(), 0, sizeof(Values)); }

        void Bias(const Row& bias) { for (auto& row : Values) Kernel::Copy(bias, row); }

        Row& operator [](const s00 perspective)
        {
            assert(perspective < PerspectiveCount);

            return Values[perspective];
        }

        const Row& operator [](const s00 perspective) const
        {
            assert(perspective < PerspectiveCount);

            return Values[perspective];
        }

    };

}

#endif
