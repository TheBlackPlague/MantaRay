#pragma once

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

        static constexpr F Apply   (F x) { return x; }
        static constexpr S ApplySum(S x) { return x; }

#ifdef SIMD

        [[clang::always_inline]]
        static SIMDVEC ApplyVector(SIMDVEC x) { return x; }

#endif

    };

    template<int Minimum, int Maximum, typename Q>
    struct Activation<ClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        static constexpr bool SIMDCompatible = true;

        template<typename T, i64 Scale>
        static constexpr T Clamp(T x)
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

        static constexpr F Apply   (F x) { return Clamp<F,                           Q::QA>(x); }
        static constexpr S ApplySum(S x) { return Clamp<S, static_cast<i64>(Q::QA) * Q::QB>(x); }

#ifdef SIMD

        [[clang::always_inline]]
        static SIMDVEC ApplyVector(SIMDVEC x)
        {
            static_assert(
                static_cast<i64>(Minimum) * Q::QA >= std::numeric_limits<F>::min() &&
                static_cast<i64>(Maximum) * Q::QA <= std::numeric_limits<F>::max()
            );

            const auto lower = SIMD<F>::From(static_cast<F>(static_cast<i64>(Minimum) * Q::QA));
            const auto upper = SIMD<F>::From(static_cast<F>(static_cast<i64>(Maximum) * Q::QA));

            return SIMD<F>::Min(upper, SIMD<F>::Max(lower, x));
        }

#endif

    };

    template<int Minimum, int Maximum, typename Q>
    struct Activation<SquaredClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        using Clip = Activation<ClippedReLU<Minimum, Maximum>, Q>;

        static constexpr bool SIMDCompatible = false;

        template<typename T, i64 Scale>
        static constexpr T Square(T x)
        {
            const i64 result = static_cast<i64>(x) * static_cast<i64>(x) / Scale;

            return static_cast<T>(std::min(result, static_cast<i64>(std::numeric_limits<T>::max())));
        }

        static constexpr F Apply   (F x) { return Square<F,                           Q::QA>(Clip::Apply   (x)); }
        static constexpr S ApplySum(S x) { return Square<S, static_cast<i64>(Q::QA) * Q::QB>(Clip::ApplySum(x)); }

    };

}
