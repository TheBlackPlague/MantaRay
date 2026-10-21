//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include "../Examples.h"

using namespace Examples;

using Wrong = Architecture<
    FeatureTransformer<Chess768, 64, Weights<i16, 255, Bound>, Biases<i16, 255, Bound>>,
    ClippedReLU<0, 1>,
    Concat,
    Concat,
    Affine<256, 1, Weights<i16, 64, Bound>, Biases<i16, 255 * 64>>,
    Output<400>
>;

static_assert(Wrong::LayerCount == 5);
