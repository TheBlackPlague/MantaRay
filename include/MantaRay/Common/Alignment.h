//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_ALIGNMENT_H
#define MANTARAY_COMMON_ALIGNMENT_H

#include "Integral.h"

namespace MantaRay
{

    constexpr inline s00 Alignment = 64;

    template<u08>
    struct Absent {};

}

#if defined(_MSC_VER)
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#endif //MANTARAY_COMMON_ALIGNMENT_H
