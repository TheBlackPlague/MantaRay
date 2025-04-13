//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#include <iostream>

#include "Backend/Kernel/ArrayAdd.h"
#include "Backend/Kernel/ArraySubAdd.h"

constexpr MantaRay::s00 N = 128 * 3;

ALIGN MantaRay::Array<MantaRay::i16, N> a = []() constexpr -> MantaRay::Array<MantaRay::i16, N>
{
    MantaRay::Array<int16_t, N> result;

    for (MantaRay::s00 i = 0; i < N; i++)
        result[i] = static_cast<MantaRay::i16>(i + (i % 2 == 0 ? 30 : -15) + (i % 3 == 0 ? 10 : -5));

    return result;
}();
ALIGN MantaRay::Array<MantaRay::i16, N> b = []() constexpr -> MantaRay::Array<MantaRay::i16, N>
{
    MantaRay::Array<MantaRay::i16, N> result;

    for (MantaRay::s00 i = 0; i < N; i++)
        result[i] = static_cast<MantaRay::i16>(i + (i % 4 == 0 ? -30 : 15) + (i % 3 == 0 ? -10 : 5));

    return result;
}();

int main()
{
    MantaRay::ArraySubAdd(a, b, b);

    for (MantaRay::s00 i = 0; i < N; i++) std::cout << a[i] << " ";
}
