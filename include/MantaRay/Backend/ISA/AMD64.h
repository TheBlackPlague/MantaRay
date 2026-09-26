//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_AMD64_H
#define MANTARAY_BACKEND_ISA_AMD64_H

#include "Portable.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace MantaRay::Backend::ISA
{

    template<typename T>
    struct AMD64 : Portable<T> {};

}

#endif
