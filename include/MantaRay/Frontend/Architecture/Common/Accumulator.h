//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACCUMULATOR_H
#define MANTARAY_ACCUMULATOR_H

#include "../../../Backend/Kernel/ArrayCopy.h"

namespace MantaRay
{

    template<QuantizedInteger T, usize N, bool Colored = true>
    class Accumulator
    {

        HWY_ALIGN Array<T, N + (N * Colored)> Internal;

        public:
        [[clang::always_inline]]
        Accumulator() { Zero(); }

        [[clang::always_inline]]
        Accumulator(const Accumulator& src) { ArrayCopy(src.Internal, Internal); }

        [[clang::always_inline]]
        void operator =(const Accumulator& src) { ArrayCopy(src.Internal, Internal); }

        [[clang::always_inline]]
        void Zero() { std::memset(Internal.data(), 0, sizeof(Array<T, N + (N * Colored)>)); }

        [[clang::always_inline]]
        void Bias(const Array<T, N>& bias)
        {
            ArrayCopy(bias, Slice<0, N>(Internal));

            if (Colored)
            ArrayCopy(bias, Slice<N, N>(Internal));
        }

        [[clang::always_inline]]
              Array<T, N>& operator [](const usize side)       requires Colored { return Slice<N>(Internal, side * N); }

        [[clang::always_inline]]
        const Array<T, N>& operator [](const usize side) const requires Colored { return Slice<N>(Internal, side * N); }

    };

} // MantaRay

#endif //MANTARAY_ACCUMULATOR_H
