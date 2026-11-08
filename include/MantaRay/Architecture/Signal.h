//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_SIGNAL_H
#define MANTARAY_ARCHITECTURE_SIGNAL_H

#include <concepts>
#include <string_view>

#include "../Common.h"

namespace MantaRay
{

    enum class Domain : u08 { Integer, Float };

    struct Signal
    {

        s00      Size         = 0;
        u08      Perspectives = 2;
        Domain   Domain       = Domain::Integer;
        Interval Range        {};
        Scale    Scale        {};
        s00      Buckets      = 1;

    };

    struct Diagnostic
    {

        bool    Ok   = true;
        Message Text {};

    };

    template<typename L>
    concept Layer = requires (const Signal& signal) {
        { L::Name            } -> std::convertible_to<std::string_view>;

        { L::Check  (signal) } -> std::same_as<Diagnostic>;
        { L::Forward(signal) } -> std::same_as<Signal    >;
    };

    constexpr inline s00 VectorBytes = 64;

    constexpr Diagnostic Fail(const Message& text) { return { .Ok = false, .Text = text }; }

    constexpr Message& operator <<(Message& message, const Scale scale)
    {
        message << scale.Numerator;

        if (scale.Denominator != 1) message << "/" << scale.Denominator;

        return message;
    }

    constexpr Message& operator <<(Message& message, const Interval interval)
    {
        return message << "[" << interval.Min << ", " << interval.Max << "]";
    }

    constexpr bool Bounded(const Interval range)
    {
        return -Detail::Saturation < range.Min && range.Max < Detail::Saturation;
    }

    constexpr s00 StorageBytes(const Interval range)
    {
        if (Fits<i08>(range) || Fits<u08>(range)) return 1;
        if (Fits<i16>(range) || Fits<u16>(range)) return 2;

        return 4;
    }

    constexpr s00 AccumulatorBytes(const Interval range) { return Fits<i16>(range) ? 2 : 4; }

}

#endif //MANTARAY_ARCHITECTURE_SIGNAL_H
