//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TRANSFORM_AFFINE_H
#define MANTARAY_ARCHITECTURE_TRANSFORM_AFFINE_H

#include <string_view>
#include <type_traits>

#include "Schedule.h"
#include "../Signal.h"
#include "../../Common/Integral.h"
#include "../../Common/Interval.h"
#include "../../Common/Message.h"

namespace MantaRay
{

    template<s00 In, s00 Out, typename W, typename B, typename S = Exact>
    struct Affine
    {

        static_assert(
            IntegerSchedule<S> || std::is_same_v<S, Dequantize>,
            "Affine schedules are Exact, DivideBy<D>, ShiftRight<K> and Dequantize"
        );

        using WeightSpec = W;
        using   BiasSpec = B;

        using Schedule = S;

        constexpr static s00  InputSize = In ;
        constexpr static s00 OutputSize = Out;

        constexpr static bool IsFloat = std::is_same_v<typename W::Type, f32>;

        constexpr static bool IsDequantize = std::is_same_v<S, Dequantize>;

        constexpr static std::string_view Name = "Affine";

        consteval static Interval SumRange(const Signal& signal) { return Repeat(signal.Range * W::Bounds, In); }

        consteval static Scale SumScale(const Signal& signal) { return signal.Scale * Scale { W::Quantization, 1 }; }

        consteval static f32 DequantizeFactor(const Signal& signal)
        {
            const auto [numerator, denominator] = SumScale(signal);

            return static_cast<f32>(static_cast<double>(denominator) / static_cast<double>(numerator));
        }

        consteval static Diagnostic CheckFloat(const Signal& signal)
        {
            if (!std::is_same_v<typename B::Type, f32>) return Fail(Message {} << "f32 weights require f32 biases");
            if (!std::is_same_v<S, Exact>             ) return Fail(Message {} << "f32 layers take no schedule");

            if (signal.Domain != Domain::Float)
                return Fail(Message {} << "f32 weights require a float input; use Dequantize on the previous Affine");

            constexpr s00 Lanes = VectorBytes / sizeof(f32);

            if (In % Lanes != 0) return Fail(Message {} << "input size " << In << " is not a multiple of " << Lanes);

            return {};
        }

        consteval static Diagnostic CheckInteger(const Signal& signal)
        {
            if (signal.Domain != Domain::Integer) return Fail(Message {} << "integer weights require an integer input");

            if (!Fits<i32>(signal.Range))
                return Fail(Message {} << "input range " << signal.Range << " does not fit i32");

            const s00 lanes = VectorBytes / StorageBytes(signal.Range);

            if (In % lanes != 0) return Fail(Message {} << "input size " << In << " is not a multiple of " << lanes);

            if (!Bounded(SumRange(signal))) return Fail(Message {} << "sum range overflows i64");

            if constexpr (IsDequantize) {
                if (!std::is_same_v<typename B::Type, f32>) return Fail(Message {} << "Dequantize requires f32 biases");
            } else {
                if (std::is_same_v<typename B::Type, f32>) return Fail(Message {} << "f32 biases require Dequantize");

                const Scale sum  = S::Apply(SumScale(signal));
                const Scale bias { .Numerator = B::Quantization, .Denominator = 1 };

                if (sum != bias) {
                    const Scale ratio = Divide(SumScale(signal), B::Quantization);

                    Message text = Message {} << "sum scale " << sum << " does not match bias scale " << bias;

                    if (ratio.Denominator == 1 && ratio.Numerator > 1)
                        text << "; use DivideBy<" << ratio.Numerator << ">";

                    return Fail(text);
                }
            }

            return {};
        }

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (signal.Perspectives != 1) return Fail(Message {} << "Affine requires a single perspective; use Concat");

            if (signal.Size != In) {
                auto message = Message {};

                message << "expected input size " << In << " but previous layer produces " << signal.Size;

                return Fail(message);
            }

            if (Out == 0) return Fail(Message {} << "output size is zero");

            if constexpr (IsFloat) return CheckFloat  (signal);
            else                   return CheckInteger(signal);
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output { .Size = Out, .Perspectives = 1, .Domain = Domain::Float };

            if constexpr (!IsFloat && !IsDequantize) {
                output.Domain = Domain::Integer;
                output.Range  = S::Apply(SumRange(signal)) + B::Bounds;
                output.Scale  = { .Numerator = B::Quantization, .Denominator = 1 };
            }

            return output;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_TRANSFORM_AFFINE_H
