//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include "../Examples.h"

using namespace Examples;

using Wrong = Perspective<760>;

static_assert(Wrong::LayerCount == 4);
