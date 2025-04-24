//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __amd64__

#ifndef MANTARAY_AMD64_H
#define MANTARAY_AMD64_H

// ReSharper disable once CppUnusedIncludeDirective
#include <immintrin.h>

#include "../Constraint.h"

namespace MantaRay
{

    template<QuantizedInteger T>
    struct AMD64 {};

} // MantaRay

#endif //MANTARAY_AMD64_H

#endif
