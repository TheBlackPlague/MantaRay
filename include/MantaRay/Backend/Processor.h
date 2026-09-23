#pragma once

#include "../Common/Alignment.h"
#include "../Common/Container.h"

#if !defined(MANTARAY_FORCE_SCALAR) && defined(__amd64__)

#include "SIMD/SSE2.h"

#define SIMD MantaRay::SSE2
#define SIMDVEC MantaRay::Vec128I

#if defined(__SSE4_1__)

#include "SIMD/SSE41.h"

#undef SIMD
#define SIMD MantaRay::SSE41

#endif

#if defined(__AVX__)

#include "SIMD/AVX.h"

#undef SIMD
#undef SIMDVEC
#define SIMD MantaRay::AVX
#define SIMDVEC MantaRay::Vec256I

#endif

#if defined(__AVX2__)

#include "SIMD/AVX2.h"

#undef SIMD
#define SIMD MantaRay::AVX2

#endif

#if defined(__AVX512F__)

#include "SIMD/AVX512F.h"

#undef SIMD
#undef SIMDVEC
#define SIMD MantaRay::AVX512F
#define SIMDVEC MantaRay::Vec512I

#endif

#if defined(__AVX512BW__)

#include "SIMD/AVX512BW.h"

#undef SIMD
#define SIMD MantaRay::AVX512BW

#endif

#endif

#ifdef ALIGN

#undef ALIGN

#endif

#define ALIGN alignas(MantaRay::Alignment)
