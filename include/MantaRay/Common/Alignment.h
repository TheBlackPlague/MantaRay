//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ALIGNMENT_H
#define MANTARAY_ALIGNMENT_H

#include "Integral.h"

namespace MantaRay
{

    constexpr inline s00 Alignment = 64;

}

#ifdef ALIGN

#undef ALIGN

#endif

#define ALIGN alignas(MantaRay::Alignment)

#endif
