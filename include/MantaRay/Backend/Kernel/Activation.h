//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ACTIVATION_H
#define MANTARAY_BACKEND_ACTIVATION_H

#include <algorithm>
#include <limits>

#include "../Processor.h"
#include "../../Architecture/Activation/ClippedReLU.h"
#include "../../Architecture/Activation/Identity.h"
#include "../../Architecture/Activation/SquaredClippedReLU.h"

namespace MantaRay::Backend
{

    template<typename A, typename Q> struct Activation;

    template<typename Q>
    struct Activation<Identity, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        static constexpr bool SIMDCompatible = true;

        static constexpr F Apply   (const F x) { return x; }
        static constexpr S ApplySum(const S x) { return x; }

        [[clang::always_inline]]
        static SIMDVEC ApplyVector(const SIMDVEC x) requires HasSIMD { return x; }

    };

    template<i32 Minimum, i32 Maximum, typename Q>
    struct Activation<ClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        static constexpr bool SIMDCompatible = true;

        template<typename T, i64 Scale>
        static constexpr T Clamp(const T x)
        {
            static_assert(
                static_cast<i64>(Minimum) * Scale >= std::numeric_limits<T>::min() &&
                static_cast<i64>(Maximum) * Scale <= std::numeric_limits<T>::max(),
                "Activation bounds must fit the quantized domain."
            );

            return std::clamp(
                x,
                static_cast<T>(static_cast<i64>(Minimum) * Scale),
                static_cast<T>(static_cast<i64>(Maximum) * Scale)
            );
        }

        static constexpr F Apply   (const F x) { return Clamp<F,                           Q::QA>(x); }
        static constexpr S ApplySum(const S x) { return Clamp<S, static_cast<i64>(Q::QA) * Q::QB>(x); }

        [[clang::always_inline]]
        static SIMDVEC ApplyVector(const SIMDVEC x) requires HasSIMD
        {
            static_assert(
                static_cast<i64>(Minimum) * Q::QA >= std::numeric_limits<F>::min() &&
                static_cast<i64>(Maximum) * Q::QA <= std::numeric_limits<F>::max()
            );

            const auto lower = SIMD<F>::From(static_cast<F>(static_cast<i64>(Minimum) * Q::QA));
            const auto upper = SIMD<F>::From(static_cast<F>(static_cast<i64>(Maximum) * Q::QA));

            return SIMD<F>::Min(upper, SIMD<F>::Max(lower, x));
        }

    };

    template<i32 Minimum, i32 Maximum, typename Q>
    struct Activation<SquaredClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        using Clip = Activation<ClippedReLU<Minimum, Maximum>, Q>;

        static constexpr bool SIMDCompatible = false;

        template<typename T, i64 Scale>
        static constexpr T Square(const T x)
        {
            const i64 result = static_cast<i64>(x) * static_cast<i64>(x) / Scale;

            return static_cast<T>(std::min(result, static_cast<i64>(std::numeric_limits<T>::max())));
        }

        static constexpr F Apply   (const F x) { return Square<F,                           Q::QA>(Clip::Apply   (x)); }
        static constexpr S ApplySum(const S x) { return Square<S, static_cast<i64>(Q::QA) * Q::QB>(Clip::ApplySum(x)); }

    };

}

#endif
