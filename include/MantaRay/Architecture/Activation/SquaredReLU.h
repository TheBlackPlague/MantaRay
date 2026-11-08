//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDRELU_H
#define MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDRELU_H

#include <string_view>

#include "ReLU.h"
#include "../Signal.h"
#include "../../Common.h"

namespace MantaRay
{

    struct SquaredReLU
    {

        constexpr static std::string_view Name = "SquaredReLU";

        consteval static Diagnostic Check(const Signal&) { return {}; }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = ReLU::Forward(signal);

            if (signal.Domain == Domain::Integer) {
                output.Range = Square(output.Range);
                output.Scale = signal.Scale * signal.Scale;
            }

            return output;
        }

        constexpr static i64 Apply(const i64 value, const Scale& scale)
        {
            const i64 positive = ReLU::Apply(value, scale);

            return positive * positive;
        }

        constexpr static f32 ApplyFloat(const f32 value)
        {
            const f32 positive = ReLU::ApplyFloat(value);

            return positive * positive;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDRELU_H
