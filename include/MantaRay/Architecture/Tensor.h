//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TENSOR_H
#define MANTARAY_ARCHITECTURE_TENSOR_H

#include <algorithm>
#include <limits>
#include <type_traits>

#include "../Common.h"

namespace MantaRay
{

    template<double Lower, double Upper>
    struct Clip
    {

        static_assert(Lower <= Upper, "Clip requires Lower <= Upper");

        constexpr static double Minimum = Lower;
        constexpr static double Maximum = Upper;

    };

    struct Transposed {};

    namespace Detail
    {

        template<typename O>
        constexpr inline bool IsClip = false;

        template<double Lower, double Upper>
        constexpr inline bool IsClip<Clip<Lower, Upper>> = true;

        constexpr i64 RoundHalfAway(const double value)
        {
            return value < 0 ? -static_cast<i64>(0.5 - value) : static_cast<i64>(value + 0.5);
        }

        template<typename T, i64 Q, typename... Options>
        consteval Interval TensorBounds()
        {
            if constexpr (std::is_same_v<T, f32>) return {};
            else {
                constexpr double Low  = std::numeric_limits<T>::min();
                constexpr double High = std::numeric_limits<T>::max();

                Interval bounds { std::numeric_limits<T>::min(), std::numeric_limits<T>::max() };

                ([&] {
                    if constexpr (IsClip<Options>) bounds = {
                        .Min = RoundHalfAway(std::clamp(Options::Minimum * Q, Low, High)),
                        .Max = RoundHalfAway(std::clamp(Options::Maximum * Q, Low, High))
                    };
                }(), ...);

                return bounds;
            }
        }

    }

    template<typename T, i64 Q, typename... Options>
    struct Tensor
    {

        static_assert(
            std::is_same_v<T, i08> || std::is_same_v<T, i16> ||
            std::is_same_v<T, i32> || std::is_same_v<T, f32>  ,
            "tensor type must be i08, i16, i32 or f32"
        );

        static_assert(Q > 0, "tensor scale must be positive");
        static_assert(Q == 1 || !std::is_same_v<T, f32>, "f32 tensors have scale 1");

        static_assert(
            ((Detail::IsClip<Options> || std::is_same_v<Options, Transposed>) && ...),
            "tensor options are Clip<Lower, Upper> and Transposed"
        );

        static_assert((s00 { Detail::IsClip<Options> } + ... + 0) <= 1, "at most one Clip per tensor");

        using Type = T;

        constexpr static i64 Quantization = Q;

        constexpr static bool IsTransposed = (std::is_same_v<Options, Transposed> || ...);
        constexpr static bool IsClipped    = (Detail::IsClip<Options            > || ...);

        constexpr static Interval Bounds = Detail::TensorBounds<T, Q, Options...>();

    };

    template<typename T, i64 Scale = 1, typename... Options>
    struct Weights : Tensor<T, Scale, Options...> {};

    template<typename T, i64 Scale = 1, typename... Options>
    struct Biases : Tensor<T, Scale, Options...> {};

}

#endif //MANTARAY_ARCHITECTURE_TENSOR_H
