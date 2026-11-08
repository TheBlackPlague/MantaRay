//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_OUTPUT_H
#define MANTARAY_ARCHITECTURE_OUTPUT_H

#include <limits>
#include <numeric>
#include <string_view>

#include "Signal.h"
#include "../Common.h"

namespace MantaRay
{

    template<i64 S>
    struct Output
    {

        constexpr static i64 OutputScale = S;

        constexpr static bool Terminal = true;

        constexpr static std::string_view Name = "Output";

        consteval static Interval Product(const Signal& signal)
        {
            return signal.Range * Interval { .Min = S * signal.Scale.Denominator, .Max = S * signal.Scale.Denominator };
        }

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (S <= 0) return Fail(Message {} << "output scale " << S << " is not positive");

            if (signal.Perspectives != 1) return Fail(Message {} << "Output requires a single perspective");

            if (signal.Size != 1)
                return Fail(Message {} << "expected a single value but previous layer produces " << signal.Size);

            if (signal.Domain == Domain::Float) return {};

            if (!Bounded(Product(signal))) return Fail(Message {} << "output product overflows i64");

            const Interval range = Divide(Product(signal), signal.Scale.Numerator);

            if (!Fits<i32>(range)) return Fail(Message {} << "output range " << range << " does not fit i32");

            return {};
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output {
                .Size         = 1,
                .Perspectives = 1,
                .Range        = { .Min = std::numeric_limits<i32>::min(), .Max = std::numeric_limits<i32>::max() }
            };

            if (signal.Domain == Domain::Integer) output.Range = Divide(Product(signal), signal.Scale.Numerator);

            return output;
        }

        constexpr static i32 Apply(const i64 value, const Scale& scale)
        {
            return static_cast<i32>(value * S * scale.Denominator / scale.Numerator);
        }

        template<Signal In>
        constexpr static i32 Apply(const i64 value)
        {
            static_assert(
                In.Domain == Domain::Integer && In.Scale.Numerator > 0 && In.Scale.Denominator > 0,
                "a compile-time Apply takes an integer signal with a positive scale"
            );

            constexpr i64 Common  = std::gcd(S * In.Scale.Denominator, In.Scale.Numerator);
            constexpr i64 Factor  = S * In.Scale.Denominator / Common;
            constexpr i64 Divisor = In.Scale.Numerator       / Common;

            constexpr bool Narrow = Fits<i32>(In.Range * Interval { .Min = Factor , .Max = Factor  }) &&
                                    Fits<i32>(           Interval { .Min = Factor , .Max = Factor  }) &&
                                    Fits<i32>(           Interval { .Min = Divisor, .Max = Divisor });

            if constexpr (Narrow) return static_cast<i32>(value) * i32 { Factor } / i32 { Divisor };

            return static_cast<i32>(value * Factor / Divisor);
        }

        constexpr static i32 ApplyFloat(const f32 value)
        {
            const f32 scaled = value * static_cast<f32>(S);

            // ReSharper disable once CppIdenticalOperandsInBinaryExpression
            if (scaled != scaled) return 0; // evil floating point

            if (scaled >=  2147483648.0F) return std::numeric_limits<i32>::max();
            if (scaled <= -2147483648.0F) return std::numeric_limits<i32>::min();

            return static_cast<i32>(scaled);
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_OUTPUT_H
