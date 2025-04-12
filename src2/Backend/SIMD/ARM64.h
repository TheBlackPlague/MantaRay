//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __aarch64__

#ifndef MANTARAY_ARM64_H
#define MANTARAY_ARM64_H

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

#endif //MANTARAY_ARM64_H

#endif
