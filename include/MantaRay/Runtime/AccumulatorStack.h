//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_ACCUMULATORSTACK_H
#define MANTARAY_RUNTIME_ACCUMULATORSTACK_H

#include <array>
#include <cassert>

#include "State.h"

namespace MantaRay::Runtime
{

    template<typename Architecture, s00 Capacity>
    class AccumulatorStack
    {

        static_assert(Capacity != 0);

        using Accumulator = State<Architecture>;

        std::array<Accumulator, Capacity> States {};

        s00 Index = 0;

        public:
        void Reset() noexcept { Index = 0; }

        [[clang::always_inline]]
        void Push()
        {
            assert(Index < Capacity - 1);

            States[Index + 1] = States[Index];

            ++Index;
        }

        [[clang::always_inline]]
        void Pop()
        {
            assert(Index != 0);

            --Index;
        }

        [[clang::always_inline]]
        void operator ++(int) { Push(); }
        [[clang::always_inline]]
        void operator --(int) {  Pop(); }

              Accumulator& operator *()       noexcept { return States[Index]; }
        const Accumulator& operator *() const noexcept { return States[Index]; }

        s00 Depth() const noexcept { return Index; }

    };

}

#endif
