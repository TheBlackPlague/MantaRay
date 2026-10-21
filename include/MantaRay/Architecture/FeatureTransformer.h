//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_FEATURETRANSFORMER_H
#define MANTARAY_ARCHITECTURE_FEATURETRANSFORMER_H

#include <string_view>
#include <type_traits>

#include "FeatureSet.h"
#include "Signal.h"
#include "../Common.h"

namespace MantaRay
{

    template<FeatureSet F, s00 Hidden, typename W, typename B>
    struct FeatureTransformer
    {

        using Features = F;

        using WeightSpec = W;
        using   BiasSpec = B;

        constexpr static s00 OutputSize = Hidden;

        constexpr static std::string_view Name = "FeatureTransformer";

        consteval static Interval Range() { return B::Bounds + Repeat(W::Bounds, F::MaxActive); }

        consteval static Diagnostic Check()
        {
            if (std::is_same_v<typename W::Type, f32> || std::is_same_v<typename B::Type, f32>)
                return Fail(Message {} << "feature transformer tensors must be integers");

            if (W::Quantization != B::Quantization) {
                auto message = Message {};

                message << "weight scale " << W::Quantization << " does not match bias scale " << B::Quantization;

                return Fail(message);
            }

            if (!Fits<i32>(Range())) return Fail(Message {} << "accumulator range " << Range() << " does not fit i32");

            const s00 lanes = VectorBytes / AccumulatorBytes(Range());

            if (Hidden == 0 || Hidden % lanes != 0)
                return Fail(Message {} << "hidden size " << Hidden << " is not a multiple of " << lanes);

            return {};
        }

        consteval static Signal Forward()
        {
            return {
                .Size         = Hidden,
                .Perspectives = 2,
                .Domain       = Domain::Integer,
                .Range        = Range(),
                .Scale        = { .Numerator = W::Quantization, .Denominator = 1 },
                .Buckets      = 1
            };
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_FEATURETRANSFORMER_H
