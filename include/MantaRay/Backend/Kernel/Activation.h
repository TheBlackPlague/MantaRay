//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_ACTIVATION_H
#define MANTARAY_BACKEND_KERNEL_ACTIVATION_H

#include <algorithm>
#include <limits>
#include <type_traits>

#include "../Native.h"
#include "../Pass/Range.h"
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

        constexpr static bool SIMDCompatible = true;

        constexpr static F Apply   (const F value) { return value; }
        constexpr static S ApplySum(const S value) { return value; }

        template<s00 Lanes>
        [[clang::always_inline]]
        static Native::Register<F, Lanes> ApplyVector(const Native::Register<F, Lanes> value)
        {
            return value;
        }

    };

    template<i32 Minimum, i32 Maximum, typename Q>
    struct Activation<ClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        constexpr static bool SIMDCompatible = true;

        template<typename T, i64 Scale>
        constexpr static T Clamp(const T value)
        {
            constexpr i64 Lower = i64 { Minimum } * Scale;
            constexpr i64 Upper = i64 { Maximum } * Scale;

            static_assert(Lower >= std::numeric_limits<T>::min() && Upper <= std::numeric_limits<T>::max());

            return std::clamp(value, static_cast<T>(Lower), static_cast<T>(Upper));
        }

        constexpr static F Apply   (const F value) { return Clamp<F,                 Q::QA>(value); }
        constexpr static S ApplySum(const S value) { return Clamp<S, i64 { Q::QA } * Q::QB>(value); }

        template<s00 Lanes>
        [[clang::always_inline]]
        static Native::Register<F, Lanes> ApplyVector(const Native::Register<F, Lanes> value)
        {
            const auto lower = Native::Broadcast<F, Lanes>(static_cast<F>(i64 { Minimum } * Q::QA));
            const auto upper = Native::Broadcast<F, Lanes>(static_cast<F>(i64 { Maximum } * Q::QA));

            return Native::Min(Native::Max(value, lower), upper);
        }

    };

    template<i32 Minimum, i32 Maximum, typename Q>
    struct Activation<SquaredClippedReLU<Minimum, Maximum>, Q>
    {

        using F = Q::FeatureType;
        using S = Q::    SumType;

        using Clip = Activation<ClippedReLU<Minimum, Maximum>, Q>;

        using Proof = Pass::SquareProof<i64 { Minimum } * Q::QA, i64 { Maximum } * Q::QA, Q::QA>;

        constexpr static bool SIMDCompatible = std::is_same_v<F, i16> && Proof::Valid;

        template<typename T, i64 Scale>
        constexpr static T Square(const T value)
        {
            const i64 normalized = i64 { value } * value / Scale;

            const i64 maximum = std::numeric_limits<T>::max();

            return static_cast<T>(std::min(normalized, maximum));
        }

        constexpr static F Apply   (const F value) { return Square<F,                 Q::QA>(Clip::Apply   (value)); }
        constexpr static S ApplySum(const S value) { return Square<S, i64 { Q::QA } * Q::QB>(Clip::ApplySum(value)); }

        template<s00 Lanes>
        [[clang::always_inline]]
        static Native::Register<F, Lanes> ApplyVector(const Native::Register<F, Lanes> value) requires SIMDCompatible
        {
            const auto clipped = Clip::ApplyVector(value);
            const auto squared = Native::MulLo(clipped, clipped);

            if (Q::QA == 1) return squared;

            const auto multiplier = Native::Broadcast<F, Lanes>(static_cast<F>(Proof::Multiplier));

            return Native::ShiftRight<Proof::Shift>(Native::MulHighUnsigned(squared, multiplier));
        }

    };

}

#endif
