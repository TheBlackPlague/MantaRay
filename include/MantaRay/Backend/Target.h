//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_TARGET_H
#define MANTARAY_BACKEND_TARGET_H

#include "ISA/AVX2.h"
#include "ISA/AVX512.h"
#include "ISA/AVX512VNNI.h"
#include "ISA/NEON.h"
#include "ISA/Scalar.h"
#include "ISA/SSE2.h"
#include "ISA/SSE41.h"

namespace MantaRay::Backend
{

#if   defined(MANTARAY_FORCE_SCALAR)
    using Native = ISA::Scalar;
#elif defined(__AVX512BW__) && defined(__AVX512VNNI__)
    using Native = ISA::AVX512VNNI;
#elif defined(__AVX512BW__)
    using Native = ISA::AVX512;
#elif defined(__AVX2__)
    using Native = ISA::AVX2;
#elif defined(__SSE4_1__)
    using Native = ISA::SSE41;
#elif defined(__SSE2__)
    using Native = ISA::SSE2;
#elif defined(__ARM_NEON) && defined(__aarch64__)
    using Native = ISA::NEON;
#else
    using Native = ISA::Scalar;
#endif

}

#endif //MANTARAY_BACKEND_TARGET_H
