//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
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

        static constexpr i32 QA = A;
        static constexpr i32 QB = B;

        static constexpr i32 OutputScale = Scale;

        using FeatureType = Feature;
        using  WeightType =  Weight;
        using     SumType =     Sum;

    };

}

#endif
