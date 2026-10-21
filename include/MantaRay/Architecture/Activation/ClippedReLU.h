//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ACTIVATION_CLIPPEDRELU_H
#define MANTARAY_ARCHITECTURE_ACTIVATION_CLIPPEDRELU_H

#include <algorithm>
#include <string_view>

#include "../Signal.h"
#include "../../Common/Integral.h"
#include "../../Common/Interval.h"
#include "../../Common/Message.h"

namespace MantaRay
{

    template<i32 Min = 0, i32 Max = 1>
    struct ClippedReLU
    {

        constexpr static i32 Minimum = Min;
        constexpr static i32 Maximum = Max;

        constexpr static std::string_view Name = "ClippedReLU";

        constexpr static Interval Bounds(const Scale& scale)
        {
            return {
                .Min = Min * scale.Numerator / scale.Denominator,
                .Max = Max * scale.Numerator / scale.Denominator
            };
        }

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (Min >= Max) return Fail(Message {} << "clamp bounds [" << Min << ", " << Max << "] are empty");

            if (signal.Domain == Domain::Float) return {};

            const Scale& scale = signal.Scale;

            if (Min * scale.Numerator % scale.Denominator != 0 || Max * scale.Numerator % scale.Denominator != 0) {
                auto message = Message {};

                message << "clamp bounds [" << Min << ", " << Max << "] times scale " << scale << " are not integers";

                return Fail(message);
            }

            return {};
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = signal;

            if (signal.Domain == Domain::Integer) {
                const auto [_, max] = Bounds(signal.Scale);

                output.Range = Clamp(signal.Range, n, max);
            }

            return output;
        }

        constexpr static i64 Apply(const i64 value, const Scale& scale)
        {
            const auto [min, max] = Bounds(scale);

            return std::clamp(value, min, max);
        }

        constexpr static f32 ApplyFloat(const f32 value) { return std::clamp(value, f32 { Min }, f32 { Max }); }

    };

}

#endif //MANTARAY_ARCHITECTURE_ACTIVATION_CLIPPEDRELU_H
