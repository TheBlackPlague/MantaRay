//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_NODE_OPERATION_H
#define MANTARAY_BACKEND_NODE_OPERATION_H

#include <array>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::Node
{

    enum class Operation
    {

        Input,
        Affine,
        Accumulate,
        Activate,
        Concat,
        Add,
        Requantize,
        Scale,
        AffineAccumulate,
        ConcatAffine,
        ActivateConcatAffine

    };

    enum class Activation { Identity, Clipped, Squared };

    enum class Domain     { Feature, Product };

    template<s00 Capacity>
    struct Instruction
    {

        Operation Code = Operation::Input;

        Domain Unit = Domain::Feature;

        std::array<s00, Capacity> Inputs {};

        s00 InputCount = 0;
        s00       Size = 0;
        s00  Parameter = 0;
        s00      State = 0;
        s00      Width = 0;

        bool FeatureAffine = false;
        bool NarrowSquare  = false;
        bool Bounded       = false;

        Activation Function = Activation::Identity;

        i32 Minimum = 0;
        i32 Maximum = 1;

        i64 Lower = 0;
        i64 Upper = 0;

        constexpr bool operator ==(const Instruction&) const = default;

    };

}

#endif
