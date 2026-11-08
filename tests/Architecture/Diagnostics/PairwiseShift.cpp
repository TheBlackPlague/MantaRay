//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include "../Examples.h"

using namespace Examples;

using Wrong = Architecture<
    FeatureTransformer<Chess768, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
    Pairwise<ClippedReLU<0, 1>, 63>,
    Concat,
    Affine<64, 1, Weights<i08, 64, Bound>, Biases<f32>, Dequantize>,
    Output<400>
>;

static_assert(Wrong::LayerCount == 4);
