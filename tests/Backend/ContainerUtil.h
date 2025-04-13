//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINERUTIL_H
#define MANTARAY_CONTAINERUTIL_H

#include <MantaRayV2/Backend/Container.h>

template<MantaRay::QuantizedInteger T, MantaRay::s00 N, T A>
constexpr inline MantaRay::Array<T, N> Generate()
{
    MantaRay::Array<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++)
        result[i] = static_cast<T>(i % 2 == 0 ? A : 0);

    return result;
}

#endif //MANTARAY_CONTAINERUTIL_H
