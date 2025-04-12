//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARM64_H
#define MANTARAY_ARM64_H

#ifdef __aarch64__

// ReSharper disable CppUnusedIncludeDirective
#include <arm_neon.h>
#include <arm_sve.h>
// ReSharper restore CppUnusedIncludeDirective

#include "Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    struct ARM64 {};

} // MantaRay

#endif

#endif //MANTARAY_ARM64_H
