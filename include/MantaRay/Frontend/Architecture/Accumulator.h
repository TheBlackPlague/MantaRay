//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACCUMULATOR_H
#define MANTARAY_ACCUMULATOR_H

#include "../../Backend/Kernel/ArrayCopy.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N, bool Colored = true>
    class Accumulator
    {

        ALIGN Array<T, N + (N * Colored)> Internal;

        public:
        Accumulator() { Zero(); }

        Accumulator(const Accumulator& src) { ArrayCopy(src.Internal, Internal); }

        inline void Zero() { std::memset(Internal.data(), 0, sizeof(Array<T, N + (N * Colored)>)); }

        inline void Bias(Array<T, N>& bias)
        {
            ArrayCopy(bias, Slice<0, N>(Internal));

            if (Colored)
            ArrayCopy(bias, Slice<N, N>(Internal));
        }

              Array<T, N>& operator [](const s00 side)       requires Colored { return Slice<N>(Internal, side * N); }
        const Array<T, N>& operator [](const s00 side) const requires Colored { return Slice<N>(Internal, side * N); }

    };

} // MantaRay

#endif //MANTARAY_ACCUMULATOR_H
