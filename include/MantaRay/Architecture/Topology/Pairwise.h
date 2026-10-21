//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TOPOLOGY_PAIRWISE_H
#define MANTARAY_ARCHITECTURE_TOPOLOGY_PAIRWISE_H

#include <string_view>

#include "../Signal.h"
#include "../../Common.h"

namespace MantaRay
{

    template<Layer A, s00 K = 0>
    struct Pairwise
    {

        static_assert(K < 63, "Pairwise requires a shift below 63");

        using Activation = A;

        constexpr static s00 Shift = K;

        constexpr static std::string_view Name = "Pairwise";

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (signal.Perspectives != 2) return Fail(Message {} << "Pairwise requires two perspectives");
            if (signal.Size % 2     != 0) return Fail(Message {} << "size " << signal.Size << " is not even");

            const s00 half  = signal.Size / 2;
            const s00 lanes = VectorBytes / AccumulatorBytes(signal.Range);

            if (half % lanes != 0)
                return Fail(Message {} << "half size " << half << " is not a multiple of " << lanes);

            return A::Check(signal);
        }

        consteval static Signal Forward(const Signal& signal)
        {
            const Signal activated = A::Forward(signal);

            Signal output = activated;

            output.Size  = signal.Size / 2;
            output.Range = ShiftDown(activated.Range * activated.Range, K);
            output.Scale = ShiftDown(activated.Scale * activated.Scale, K);

            return output;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_TOPOLOGY_PAIRWISE_H
