//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_INTERVAL_H
#define MANTARAY_COMMON_INTERVAL_H

#include <algorithm>
#include <concepts>
#include <limits>

#include "Integral.h"

namespace MantaRay
{

    struct Interval
    {

        i64 Min = 0;
        i64 Max = 0;

        constexpr bool operator ==(const Interval&) const = default;

    };

    struct Scale
    {

        i64 Numerator   = 1;
        i64 Denominator = 1;

        constexpr bool operator ==(const Scale&) const = default;

    };

    namespace Detail
    {

        using i128 = __int128;

        constexpr inline i64 Saturation = i64 { 1 } << 62;

        constexpr i64 Saturate(const i128 value)
        {
            if (value >  Saturation) return  Saturation;
            if (value < -Saturation) return -Saturation;

            return static_cast<i64>(value);
        }

        constexpr i64 Narrow(const i128 value)
        {
            if (value > std::numeric_limits<i64>::max() || value < std::numeric_limits<i64>::min()) __builtin_trap();

            return static_cast<i64>(value);
        }

        constexpr i128 Gcd(i128 left, i128 right)
        {
            while (right != 0) {
                const i128 rest = left % right;

                left  = right;
                right = rest ;
            }

            return left;
        }

        constexpr Scale Reduce(i128 numerator, i128 denominator)
        {
            if (denominator == 0) __builtin_trap();

            if (denominator < 0) {
                numerator   = -numerator  ;
                denominator = -denominator;
            }

            const i128 divisor = Gcd(numerator < 0 ? -numerator : numerator, denominator);

            return { .Numerator = Narrow(numerator / divisor), .Denominator = Narrow(denominator / divisor) };
        }

    }

    constexpr Interval operator +(const Interval left, const Interval right)
    {
        return {
            .Min = Detail::Saturate(Detail::i128 { left.Min } + right.Min),
            .Max = Detail::Saturate(Detail::i128 { left.Max } + right.Max)
        };
    }

    constexpr Interval operator *(const Interval left, const Interval right)
    {
        const Detail::i128 a = Detail::i128 { left.Min } * right.Min;
        const Detail::i128 b = Detail::i128 { left.Min } * right.Max;
        const Detail::i128 c = Detail::i128 { left.Max } * right.Min;
        const Detail::i128 d = Detail::i128 { left.Max } * right.Max;

        return { .Min = Detail::Saturate(std::min({ a, b, c, d })), .Max = Detail::Saturate(std::max({ a, b, c, d })) };
    }

    constexpr Interval Repeat(const Interval interval, const s00 count)
    {
        const Detail::i128 low  = Detail::i128 { interval.Min } * count;
        const Detail::i128 high = Detail::i128 { interval.Max } * count;

        return {
            .Min = Detail::Saturate(std::min<Detail::i128>(low , 0)),
            .Max = Detail::Saturate(std::max<Detail::i128>(high, 0))
        };
    }

    constexpr Interval Clamp(const Interval interval, const i64 lower, const i64 upper)
    {
        return { .Min = std::clamp(interval.Min, lower, upper), .Max = std::clamp(interval.Max, lower, upper) };
    }

    constexpr Interval Square(const Interval interval)
    {
        const Detail::i128 low  = Detail::i128 { interval.Min } * interval.Min;
        const Detail::i128 high = Detail::i128 { interval.Max } * interval.Max;

        const bool straddles = interval.Min <= 0 && interval.Max >= 0;

        return {
            .Min = Detail::Saturate(straddles ? 0 : std::min(low, high)),
            .Max = Detail::Saturate(                std::max(low, high))
        };
    }

    constexpr Interval ShiftDown(const Interval interval, const s00 shift)
    {
        const s00 amount = std::min<s00>(shift, 63);

        return { .Min = interval.Min >> amount, .Max = interval.Max >> amount };
    }

    constexpr Interval Divide(const Interval interval, const i64 divisor)
    {
        const i64 low  = interval.Min / divisor;
        const i64 high = interval.Max / divisor;

        return { .Min = std::min(low, high), .Max = std::max(low, high) };
    }

    constexpr Interval Hull(const Interval left, const Interval right)
    {
        return { .Min = std::min(left.Min, right.Min), .Max = std::max(left.Max, right.Max) };
    }

    template<std::integral T>
    constexpr bool Fits(const Interval interval)
    {
        return Detail::i128 { interval.Min } >= std::numeric_limits<T>::min() &&
               Detail::i128 { interval.Max } <= std::numeric_limits<T>::max() ;
    }

    constexpr Scale operator *(const Scale left, const Scale right)
    {
        return Detail::Reduce(
            Detail::i128 { left.Numerator   } * right.Numerator  ,
            Detail::i128 { left.Denominator } * right.Denominator
        );
    }

    constexpr Scale Divide(const Scale scale, const i64 divisor)
    {
        return Detail::Reduce(scale.Numerator, Detail::i128 { scale.Denominator } * divisor);
    }

    constexpr Scale ShiftDown(const Scale scale, const s00 shift)
    {
        return Detail::Reduce(scale.Numerator, Detail::i128 { scale.Denominator } << shift);
    }

}

#endif //MANTARAY_COMMON_INTERVAL_H
