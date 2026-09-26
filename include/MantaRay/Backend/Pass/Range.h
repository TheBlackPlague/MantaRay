//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_PASS_RANGE_H
#define MANTARAY_BACKEND_PASS_RANGE_H

#include "../../Common/Integral.h"

namespace MantaRay::Backend::Pass
{

    struct DivisionProof
    {

        bool Valid = false;

        u16 Multiplier = 0;
        s00      Shift = 0;

    };

    constexpr DivisionProof ProveDivision(const i64 maximum, const i64 divisor)
    {
        if (maximum < 0 || maximum > 32767 || divisor <= 0) return {};

        if (divisor == 1) return { .Valid = true, .Multiplier = 0, .Shift = 0 };

        for (s00 shift = 16; shift < 32; shift++) {
            const i64 unit = i64{1} << shift;
            const i64 multiplier = (unit + divisor - 1) / divisor;

            if (multiplier > 65535) break;

            const i64 error = multiplier * divisor - unit;
            const i64 limit = maximum / divisor * error + (divisor - 1) * multiplier;

            if (limit < unit)
                return { .Valid = true, .Multiplier = static_cast<u16>(multiplier), .Shift = shift - 16 };
        }

        return {};
    }

    template<i64 Lower, i64 Upper, i64 Scale>
    struct SquareProof
    {

        constexpr static bool Bounded = Scale > 0 && Lower >= 0 && Upper >= Lower && Upper <= 181;

        constexpr static DivisionProof Division = ProveDivision(Bounded ? Upper * Upper : -1, Scale);

        constexpr static bool Valid = Bounded && Division.Valid;

        constexpr static u16 Multiplier = Division.Multiplier;
        constexpr static s00 Shift      = Division.Shift     ;

    };

}

#endif
