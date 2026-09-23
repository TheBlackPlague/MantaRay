//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#pragma once

#include "../Common/Constraint.h"

namespace MantaRay
{

    template<
        int A,
        int B,
        int Scale,
        QuantizedInteger Feature = i16,
        QuantizedInteger Weight = i16,
        QuantizedInteger Sum = i32
    >
    struct Quantization
    {
        static constexpr int QA = A;
        static constexpr int QB = B;

        static constexpr int OutputScale = Scale;

        using FeatureType = Feature;
        using  WeightType =  Weight;
        using     SumType =     Sum;

    };

}
