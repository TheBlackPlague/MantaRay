//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACCUMULATOR_H
#define MANTARAY_ACCUMULATOR_H

#include "../../../Backend/Kernel/ArrayCopy.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N, bool Colored = true>
    class Accumulator
    {

        using Storage = NArray<T, 1 + Colored, N>;

        static_assert(
            sizeof(Storage) == sizeof(T) * N * (1 + Colored),
            "Accumulator storage must remain tightly packed."
        );

        ALIGN Storage Internal;

        public:
        [[clang::always_inline]]
        Accumulator() { Zero(); }

        [[clang::always_inline]]
        Accumulator(const Accumulator& src) { ArrayCopy(src.Internal, Internal); }

        [[clang::always_inline]]
        void operator =(const Accumulator& src) { ArrayCopy(src.Internal, Internal); }

        [[clang::always_inline]]
        void Zero() { std::memset(&Internal, 0, sizeof Internal); }

        [[clang::always_inline]]
        void Bias(const Array<T, N>& bias)
        {
            ArrayCopy(bias, Internal[0]);

            if (Colored)
            ArrayCopy(bias, Internal[1]);
        }

        [[clang::always_inline]]
              Array<T, N>& operator [](const s00 side)       requires Colored { assert(side < 2); return Internal[side]; }

        [[clang::always_inline]]
        const Array<T, N>& operator [](const s00 side) const requires Colored { assert(side < 2); return Internal[side]; }

    };

} // MantaRay

#endif //MANTARAY_ACCUMULATOR_H
