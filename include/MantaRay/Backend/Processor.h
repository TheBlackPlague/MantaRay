//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_PROCESSOR_H
#define MANTARAY_PROCESSOR_H

#include "../Common/Alignment.h"
#include "../Common/Container.h"

#if !defined(MANTARAY_FORCE_SCALAR) && defined(__amd64__)

#if defined(__AVX512BW__)
#include "SIMD/AVX512BW.h"
#elif defined(__AVX512F__)
#include "SIMD/AVX512F.h"
#elif defined(__AVX2__)
#include "SIMD/AVX2.h"
#elif defined(__AVX__)
#include "SIMD/AVX.h"
#elif defined(__SSE4_1__)
#include "SIMD/SSE41.h"
#else
#include "SIMD/SSE2.h"
#endif

namespace MantaRay
{

    constexpr bool HasSIMD = true;

#if defined(__AVX512BW__)
    template<QuantizedInteger T>
    using SIMD = AVX512BW<T>;
#elif defined(__AVX512F__)
    template<QuantizedInteger T>
    using SIMD = AVX512F<T>;
#elif defined(__AVX2__)
    template<QuantizedInteger T>
    using SIMD = AVX2<T>;
#elif defined(__AVX__)
    template<QuantizedInteger T>
    using SIMD = AVX<T>;
#elif defined(__SSE4_1__)
    template<QuantizedInteger T>
    using SIMD = SSE41<T>;
#else
    template<QuantizedInteger T>
    using SIMD = SSE2<T>;
#endif

    using SIMDVEC = decltype(SIMD<i32>::From(i32 {}));

}

#else

namespace MantaRay
{

    constexpr bool HasSIMD = false;

    template<QuantizedInteger _>
    struct SIMD {};

    using SIMDVEC = Array<i08, Alignment>;

}

#endif

#endif //MANTARAY_PROCESSOR_H
