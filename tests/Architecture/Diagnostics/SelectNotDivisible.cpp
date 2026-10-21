//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include "../Examples.h"

using namespace Examples;

using Wrong = Architecture<
    FeatureTransformer<MirroredKingBuckets<Layout4>, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
    SquaredClippedReLU<0, 1>,
    Concat,
    Affine<128, 7, Weights<i16, 64, Bound, Transposed>, Biases<i16, 255 * 64>, DivideBy<255>>,
    Select<MaterialCount<8>>,
    Output<400>
>;

static_assert(Wrong::LayerCount == 5);
