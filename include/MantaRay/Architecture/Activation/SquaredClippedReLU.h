//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDCLIPPEDRELU_H
#define MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDCLIPPEDRELU_H

#include <string_view>

#include "ClippedReLU.h"
#include "../Signal.h"
#include "../../Common.h"

namespace MantaRay
{

    template<i32 Min = 0, i32 Max = 1>
    struct SquaredClippedReLU
    {

        using Clipped = ClippedReLU<Min, Max>;

        constexpr static i32 Minimum = Min;
        constexpr static i32 Maximum = Max;

        constexpr static std::string_view Name = "SquaredClippedReLU";

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (Min < 0) return Fail(Message {} << "lower clamp bound " << Min << " is negative");

            return Clipped::Check(signal);
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = Clipped::Forward(signal);

            if (signal.Domain == Domain::Integer) {
                output.Range = Square(output.Range);
                output.Scale = signal.Scale * signal.Scale;
            }

            return output;
        }

        constexpr static i64 Apply(const i64 value, const Scale& scale)
        {
            const i64 clamped = Clipped::Apply(value, scale);

            return clamped * clamped;
        }

        constexpr static f32 ApplyFloat(const f32 value)
        {
            const f32 clamped = Clipped::ApplyFloat(value);

            return clamped * clamped;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_ACTIVATION_SQUAREDCLIPPEDRELU_H
