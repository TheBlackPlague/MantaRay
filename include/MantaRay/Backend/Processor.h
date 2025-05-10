//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_PROCESSOR_H
#define MANTARAY_PROCESSOR_H

#define ALIGN

#ifdef __amd64__

#ifdef __SSE2__

#include "SIMD/SSE2.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::SSE2

#ifdef SIMDVEC
#undef SIMDVEC
#endif
#define SIMDVEC MantaRay::Vec128I

#endif

#ifdef __SSE4_1__

#include "SIMD/SSE41.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::SSE41

#endif

#ifdef __AVX__

#include "SIMD/AVX.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::AVX

#ifdef SIMDVEC
#undef SIMDVEC
#endif
#define SIMDVEC MantaRay::Vec256I

#endif

#ifdef __AVX2__

#include "SIMD/AVX2.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::AVX2

#endif

#ifdef __AVX512F__

#include "SIMD/AVX512F.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::AVX512F

#ifdef SIMDVEC
#undef SIMDVEC
#endif
#define SIMDVEC MantaRay::Vec512I

#endif

#ifdef __AVX512BW__

#include "SIMD/AVX512BW.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::AVX512BW

#endif

#endif

#ifdef __aarch64__

#ifdef __ARM_NEON__

#include "SIMD/NEON.h"

#ifdef SIMD
#undef SIMD
#endif
#define SIMD MantaRay::NEON

#ifdef SIMDVEC
#undef SIMDVEC
#undef SIMDVEC_EXTEND
#endif

#define SIMDVEC MantaRay::Vec128I
#define SIMDVEC_EXTEND MantaRay::Vec128IE

#endif

#endif

#endif //MANTARAY_PROCESSOR_H
