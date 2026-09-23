#pragma once

#include <array>
#include <cassert>

#include "../Backend/Storage/Accumulator.h"

namespace MantaRay::Runtime
{

    template<class Architecture, s00 Capacity>
    class AccumulatorStack
    {
        static_assert(Capacity > 0);

        using Accumulator = Backend::Accumulator<Architecture>;

        std::array<Accumulator, Capacity> States {};

        s00 Index = 0;

        public:
        void Reset() noexcept { Index = 0; }

        [[clang::always_inline]]
        void Push() { assert(Index + 1 < Capacity); States[Index + 1] = States[Index]; Index++; }

        [[clang::always_inline]]
        void Pop() { assert(Index > 0); --Index; }

        [[clang::always_inline]]
        void operator ++(int) { Push(); }

        [[clang::always_inline]]
        void operator --(int) { Pop(); }

              Accumulator& operator *()       noexcept { return States[Index]; }
        const Accumulator& operator *() const noexcept { return States[Index]; }

        s00 Depth() const noexcept { return Index; }

    };

}
