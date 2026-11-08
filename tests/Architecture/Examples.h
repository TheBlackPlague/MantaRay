//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_EXAMPLES_H
#define MANTARAY_ARCHITECTURE_EXAMPLES_H

#include <array>

#include <MantaRay/Architecture/Architecture.h>
#include <MantaRay/Architecture/FeatureTransformer.h>
#include <MantaRay/Architecture/Output.h>
#include <MantaRay/Architecture/OutputBuckets.h>
#include <MantaRay/Architecture/Tensor.h>
#include <MantaRay/Architecture/Activation/ClippedReLU.h>
#include <MantaRay/Architecture/Activation/ReLU.h>
#include <MantaRay/Architecture/Activation/SquaredClippedReLU.h>
#include <MantaRay/Architecture/Activation/SquaredReLU.h>
#include <MantaRay/Architecture/FeatureSet/Chess768.h>
#include <MantaRay/Architecture/FeatureSet/KingBuckets.h>
#include <MantaRay/Architecture/Topology/Concat.h>
#include <MantaRay/Architecture/Topology/Pairwise.h>
#include <MantaRay/Architecture/Topology/Select.h>
#include <MantaRay/Architecture/Transform/Affine.h>
#include <MantaRay/Architecture/Transform/Schedule.h>

namespace Examples
{

    using namespace MantaRay;

    constexpr std::array<u08, 32> Layout4 { 0, 0, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  3, 3, 3, 3,
                                            3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3 };

    using Bound = Clip<-1.98, 1.98>;

    template<s00 Hidden>
    using Perspective = Architecture<
        FeatureTransformer<Chess768, Hidden, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        ClippedReLU<0, 1>,
        Concat,
        Affine<Hidden * 2, 1, Weights<i16, 64, Bound>, Biases<i16, 255 * 64>>,
        Output<400>
    >;

    using Starshard = Perspective<256>;
    using Aurora    = Perspective<384>;
    using Flare     = Perspective<768>;

    template<s00 Hidden>
    using SquaredPerspective = Architecture<
        FeatureTransformer<Chess768, Hidden, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        SquaredClippedReLU<0, 1>,
        Concat,
        Affine<Hidden * 2, 1, Weights<i16, 64, Bound>, Biases<i16, 255 * 64>, DivideBy<255>>,
        Output<400>
    >;

    using SimpleSCReLU = SquaredPerspective<64>;
    using WideSCReLU   = SquaredPerspective<768>;

    template<typename R, i64 QA, i64 QB, typename B, typename S = Exact>
    using Rectified = Architecture<
        FeatureTransformer<Chess768, 64, Weights<i16, QA, Bound>, Biases<i16, QA, Bound>>,
        R,
        Concat,
        Affine<128, 1, Weights<i16, QB, Bound>, B, S>,
        Output<400>
    >;

    using SquareFirstSCReLU = Rectified<SquaredClippedReLU<0, 1>, 181,  128, Biases<i16, 181 *  128>, DivideBy<181>>;
    using WidenedSCReLU     = Rectified<SquaredClippedReLU<0, 1>, 255, 8000, Biases<i32, 255 * 8000>, DivideBy<255>>;
    using LinearReLU        = Rectified<ReLU                    , 255,   64, Biases<i16, 255 *   64>               >;
    using WidenedSqrReLU    = Rectified<SquaredReLU             ,  32,   64, Biases<i16,  32 *   64>, DivideBy< 32>>;

    using OutputBucketed = Architecture<
        FeatureTransformer<Chess768, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        SquaredClippedReLU<0, 1>,
        Concat,
        Affine<128, 8, Weights<i16, 64, Bound, Transposed>, Biases<i16, 255 * 64>, DivideBy<255>>,
        Select<MaterialCount<8>>,
        Output<400>
    >;

    using KingBucketed = Architecture<
        FeatureTransformer<MirroredKingBuckets<Layout4>, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        SquaredClippedReLU<0, 1>,
        Concat,
        Affine<128, 8, Weights<i16, 64, Bound, Transposed>, Biases<i16, 255 * 64>, DivideBy<255>>,
        Select<MaterialCount<8>>,
        Output<400>
    >;

    using MultiLayer = Architecture<
        FeatureTransformer<MirroredKingBuckets<Layout4>, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        Pairwise<ClippedReLU<0, 1>>,
        Concat,
        Affine<64, 8 * 16, Weights<i08, 64, Bound, Transposed>, Biases<f32>, Dequantize>,
        Select<MaterialCount<8>>,
        SquaredClippedReLU<0, 1>,
        Affine<16, 8 * 32, Weights<f32, 1, Transposed>, Biases<f32>>,
        Select<MaterialCount<8>>,
        SquaredClippedReLU<0, 1>,
        Affine<32, 8, Weights<f32, 1, Transposed>, Biases<f32>>,
        Select<MaterialCount<8>>,
        Output<400>
    >;

    using PackedPairwise = Architecture<
        FeatureTransformer<Chess768, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        Pairwise<ClippedReLU<0, 1>, 9>,
        Concat,
        Affine<64, 16, Weights<i08, 64, Bound>, Biases<f32>, Dequantize>,
        SquaredClippedReLU<0, 1>,
        Affine<16, 1, Weights<f32>, Biases<f32>>,
        Output<400>
    >;

    using ShiftedPairwise = Architecture<
        FeatureTransformer<Chess768, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
        Pairwise<ClippedReLU<0, 1>, 9>,
        Concat,
        Affine<64, 1, Weights<i16, 512, Bound>, Biases<i16, 255>, DivideBy<255>>,
        Output<400>
    >;

    using WideAccumulator = Architecture<
        FeatureTransformer<Chess768, 32, Weights<i16, 256>, Biases<i16, 256>>,
        SquaredClippedReLU<0, 1>,
        Concat,
        Affine<64, 1, Weights<i16, 64, Bound>, Biases<i32, 256 * 64>, ShiftRight<8>>,
        Output<400>
    >;

}

#endif //MANTARAY_ARCHITECTURE_EXAMPLES_H
