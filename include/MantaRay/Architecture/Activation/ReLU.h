//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ACTIVATION_RELU_H
#define MANTARAY_ARCHITECTURE_ACTIVATION_RELU_H

#include <algorithm>
#include <limits>
#include <string_view>

#include "../Signal.h"
#include "../../Common.h"

namespace MantaRay
{

    struct ReLU
    {

        constexpr static std::string_view Name = "ReLU";

        consteval static Diagnostic Check(const Signal&) { return {}; }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = signal;

            if (signal.Domain == Domain::Integer)
                output.Range = Clamp(signal.Range, 0, std::numeric_limits<i64>::max());

            return output;
        }

        constexpr static i64 Apply(const i64 value, const Scale&) { return std::max<i64>(value, 0); }

        constexpr static f32 ApplyFloat(const f32 value) { return std::max(value, 0.0F); }

    };

}

#endif //MANTARAY_ARCHITECTURE_ACTIVATION_RELU_H
