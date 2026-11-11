//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_NETWORKUTIL_H
#define MANTARAY_RUNTIME_NETWORKUTIL_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <MantaRay/MantaRay.h>

using namespace MantaRay;

class Random
{

    u64 State;

    public:
    explicit constexpr Random(const u64 seed) noexcept : State(seed) {}

    constexpr u64 Next() noexcept
    {
        u64 value = State += 0x9E3779B97F4A7C15;

        value = (value ^ value >> 30) * 0xBF58476D1CE4E5B9;
        value = (value ^ value >> 27) * 0x94D049BB133111EB;

        return value ^ value >> 31;
    }

};

template<auto D, u16 I>
void Append(std::vector<std::byte>& bytes, Random& random)
{
    using M = Model<D>;
    using V = IO::Tensors<M>::template Element<I>;
    using U = std::make_unsigned_t<V>;

    constexpr IR::Tensor Spec   = M::Graph.Tensors[I];
    constexpr Interval   Bounds = IR::BoundsOf(Spec);

    const u64 span = static_cast<u64>(Bounds.Max - Bounds.Min) + 1;

    for (s00 i = 0; i < IR::ElementsOf(Spec); i++) {
        const U bits = static_cast<U>(Bounds.Min + static_cast<i64>(random.Next() % span));

        for (s00 j = 0; j < sizeof(U); j++) bytes.push_back(static_cast<std::byte>(bits >> 8 * j));
    }
}

template<auto D>
std::vector<std::byte> RandomBytes(Random& random)
{
    std::vector<std::byte> bytes;

    [&]<u16... I>(const std::integer_sequence<u16, I...>) {
        (Append<D, I>(bytes, random), ...);
    }(std::make_integer_sequence<u16, Model<D>::Graph.TensorCount> {});

    return bytes;
}

#endif //MANTARAY_RUNTIME_NETWORKUTIL_H
