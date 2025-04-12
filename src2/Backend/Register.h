//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_REGISTER_H
#define MANTARAY_REGISTER_H

#include <immintrin.h>

namespace MantaRay
{

#ifdef __SSE2__
    // 128-bit integer register
    using Vec128I = __m128i;
#endif

#ifdef __AVX__
    // 256-bit integer register
    using Vec256I = __m256i;
#endif

#ifdef __AVX512F__
    // 512-bit integer register
    using Vec512I = __m512i;
#endif

} // MantaRay

#endif //MANTARAY_REGISTER_H
