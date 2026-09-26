//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifdef __amd64__

#ifndef MANTARAY_AMD64_H
#define MANTARAY_AMD64_H

// ReSharper disable once CppUnusedIncludeDirective
#include <immintrin.h>

#include "../../Common/Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger _>
    struct AMD64 {};

} // MantaRay

#endif //MANTARAY_AMD64_H

#endif
