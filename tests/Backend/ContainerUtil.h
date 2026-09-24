//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINERUTIL_H
#define MANTARAY_CONTAINERUTIL_H

#include <MantaRay/Common/Container.h>

template<MantaRay::QuantizedInteger T, MantaRay::s00 N, T A>
constexpr MantaRay::Array<T, N> Generate()
{
    ALIGN MantaRay::Array<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++) result[i] = static_cast<T>(i % 2 == 0 ? A : 0);

    return result;
}

template<MantaRay::QuantizedInteger T, MantaRay::s00 N, MantaRay::s00 M, T A>
constexpr MantaRay::NArray<T, N, M> Generate()
{
    ALIGN MantaRay::NArray<T, N, M> result;

    for (MantaRay::s00 i = 0; i < N; i++)
    for (MantaRay::s00 j = 0; j < M; j++) result[i][j] = static_cast<T>((i * M + j) % 2 == 0 ? A : 0);

    return result;
}

#endif //MANTARAY_CONTAINERUTIL_H
