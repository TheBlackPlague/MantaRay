//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_FEATURESET_CHESS768_H
#define MANTARAY_ARCHITECTURE_FEATURESET_CHESS768_H

#include "../FeatureSet.h"
#include "../../Common/Integral.h"

namespace MantaRay
{

    struct Chess768
    {

        constexpr static s00 InputSize = 768;
        constexpr static s00 MaxActive =  32;
        constexpr static s00 KeyCount  =   1;

        constexpr static u08 Key(const u08, const u08) { return 0; }

        constexpr static u32 Index(const u08 perspective, const u08, const u16 code)
        {

            const u32 side   = (code >>  9) ^       perspective ;
            const u32 piece  = (code >>  6) &   7               ;
            const u32 square = (code &  63) ^ (56 * perspective);

            return side * 384 + piece * 64 + square;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_FEATURESET_CHESS768_H
