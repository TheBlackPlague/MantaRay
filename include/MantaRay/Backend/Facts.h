//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_FACTS_H
#define MANTARAY_BACKEND_FACTS_H

#include <algorithm>

#include "../Common/Integral.h"
#include "../Common/Interval.h"

namespace MantaRay::Backend
{

    struct DotPlan
    {

        s00  Accumulators = 1;
        bool WideReduce   = false;

    };

    consteval bool LizardExact(const Interval x, const Interval w)
    {
        return Fits<i16>(x) && Fits<i16>(w) && Fits<i16>(x * w);
    }

    consteval bool SquareFitsI16(const Interval x) { return Fits<i16>(x) && Fits<i16>(Square(x)); }

    consteval bool PackusIsCReLU(const Interval clamp) { return clamp == Interval { .Min = 0, .Max = 255 }; }

    consteval bool FitsI8(const Interval x) { return Fits<i08>(x); }

    consteval bool OffsetI16(const Interval x) { return x.Max - x.Min <= 65535; }

    consteval bool MaddExact(const Interval x, const Interval w)
    {
        return Fits<i16>(x) && Fits<i16>(w) && Fits<i32>(Repeat(x * w, 2));
    }

    consteval bool MaddubsExact(const Interval u, const Interval w)
    {
        return Fits<u08>(u) && Fits<i08>(w) && Fits<i16>(Repeat(u * w, 2));
    }

    template<typename ISA>
    consteval DotPlan PlanDot(const Interval product, const s00 terms)
    {
        constexpr s00 Lanes = ISA::Bytes / sizeof(i32);

        s00 accumulators = 1;

        while (!Fits<i32>(Repeat(product, (terms + Lanes * accumulators - 1) / (Lanes * accumulators))) &&
               Lanes * accumulators < terms) accumulators *= 2;

        return {
            .Accumulators = std::max(ISA::DotChains, accumulators),
            .WideReduce   = !Fits<i32>(Repeat(product, terms))
        };
    }

}

#endif //MANTARAY_BACKEND_FACTS_H
