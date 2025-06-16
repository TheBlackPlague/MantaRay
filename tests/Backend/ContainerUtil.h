//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINERUTIL_H
#define MANTARAY_CONTAINERUTIL_H

#include <MantaRay/Backend/Base.h>

template<MantaRay::QuantizedInteger T, MantaRay::usize N, T A>
constexpr MantaRay::Array<T, N> Generate()
{
    HWY_ALIGN MantaRay::Array<T, N> result;

    for (MantaRay::usize i = 0; i < N; i++)
        result[i] = static_cast<T>(i % 2 == 0 ? A : 0);

    return result;
}

#endif //MANTARAY_CONTAINERUTIL_H
