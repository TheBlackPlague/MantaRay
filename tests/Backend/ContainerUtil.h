//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_CONTAINERUTIL_H
#define MANTARAY_BACKEND_CONTAINERUTIL_H

#include <algorithm>
#include <array>

#include <MantaRay/Common.h>

template<typename T, MantaRay::s00... N>
struct alignas(MantaRay::Alignment) Guarded
{

    constexpr static MantaRay::s00 GuardSize = MantaRay::Alignment / sizeof(T);

    constexpr static T Sentinel = static_cast<T>(0x5A5A5A5A);

    MantaRay::Array<T, N...    > Values {};
    MantaRay::Array<T, GuardSize> Guard  {};

    constexpr Guarded() { Guard.fill(Sentinel); }

          auto& operator [](const MantaRay::s00 i)       { return Values[i]; }
    const auto& operator [](const MantaRay::s00 i) const { return Values[i]; }

    [[nodiscard]]
    bool Intact() const { return std::ranges::all_of(Guard, [](const T value) { return value == Sentinel; }); }

};

template<typename T, MantaRay::s00 N, T A>
constexpr Guarded<T, N> Generate()
{
    Guarded<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++) result[i] = static_cast<T>(i % 2 == 0 ? A : 0);

    return result;
}

template<typename T, MantaRay::s00 N, T A>
constexpr Guarded<T, N> Ramp()
{
    Guarded<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++) result[i] = static_cast<T>(A + static_cast<T>(i % 128));

    return result;
}

#endif //MANTARAY_BACKEND_CONTAINERUTIL_H
