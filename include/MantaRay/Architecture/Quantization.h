//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_QUANTIZATION_H
#define MANTARAY_QUANTIZATION_H

#include "../Common/Constraint.h"

namespace MantaRay
{

    template<
        i32 A,
        i32 B,
        i32 Scale,
        QuantizedInteger Feature = i16,
        QuantizedInteger  Weight = i16,
        QuantizedInteger     Sum = i32
    >
    struct Quantization
    {

        constexpr static i32 QA = A;
        constexpr static i32 QB = B;

        constexpr static i32 OutputScale = Scale;

        using FeatureType = Feature;
        using  WeightType =  Weight;
        using     SumType =     Sum;

    };

}

#endif
