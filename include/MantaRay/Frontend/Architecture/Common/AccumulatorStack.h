//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACCUMULATORSTACK_H
#define MANTARAY_ACCUMULATORSTACK_H

#include "Accumulator.h"

namespace MantaRay
{

    template<QuantizedInteger I, usize HiddenSize, usize AccumulatorStackSize>
    class AccumulatorStack
    {

        static_assert(AccumulatorStackSize > 0, "The accumulator stack size cannot be zero.");

        using Accumulator = Accumulator<I, HiddenSize>;
        using Storage = std::array<Accumulator, AccumulatorStackSize>;

        Storage Internal;
        usize    Pointer;

        public:
        AccumulatorStack()
        {
            const Accumulator accumulator;
            std::fill(std::begin(Internal), std::end(Internal), accumulator);

            Pointer = 0;
        }

        [[clang::always_inline]]
        void Reset() { Pointer = 0; }

        [[clang::always_inline]]
        void operator ++(int)
        {
            Internal[Pointer + 1] = Internal[Pointer];
            Pointer++;

            assert(Pointer < AccumulatorStackSize);
        }

        [[clang::always_inline]]
        void operator --(int)
        {
            assert(Pointer > 0);

            Pointer--;
        }

              Accumulator& operator *()       { return Internal[Pointer]; }
        const Accumulator& operator *() const { return Internal[Pointer]; }

    };

} // MantaRay

#endif //MANTARAY_ACCUMULATORSTACK_H
