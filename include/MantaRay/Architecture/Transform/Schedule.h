//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TRANSFORM_SCHEDULE_H
#define MANTARAY_ARCHITECTURE_TRANSFORM_SCHEDULE_H

#include <concepts>

#include "../../Common/Integral.h"
#include "../../Common/Interval.h"

namespace MantaRay
{

    struct Exact
    {

        constexpr static Interval Apply(const Interval range) { return range; }
        constexpr static Scale    Apply(const Scale    scale) { return scale; }
        constexpr static i64      Apply(const i64      value) { return value; }

    };

    template<i64 D>
    struct DivideBy
    {

        static_assert(D > 0, "DivideBy requires a positive divisor");

        constexpr static i64 Divisor = D;

        constexpr static Interval Apply(const Interval range) { return Divide(range,  D); }
        constexpr static Scale    Apply(const Scale    scale) { return Divide(scale,  D); }
        constexpr static i64      Apply(const i64      value) { return        value / D ; }

    };

    template<s00 K>
    struct ShiftRight
    {

        static_assert(K < 63, "ShiftRight requires a shift below 63");

        constexpr static s00 Shift = K;

        constexpr static Interval Apply(const Interval range) { return ShiftDown(range,   K); }
        constexpr static Scale    Apply(const Scale    scale) { return ShiftDown(scale,   K); }
        constexpr static i64      Apply(const i64      value) { return           value >> K ; }

    };

    struct Dequantize {};

    template<typename S>
    concept IntegerSchedule = requires (const Interval range, const Scale scale, const i64 value) {
        { S::Apply(range) } -> std::same_as<Interval>;
        { S::Apply(scale) } -> std::same_as<Scale   >;
        { S::Apply(value) } -> std::same_as<i64     >;
    };

}

#endif //MANTARAY_ARCHITECTURE_TRANSFORM_SCHEDULE_H
