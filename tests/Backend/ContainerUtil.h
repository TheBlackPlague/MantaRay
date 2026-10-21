//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_CONTAINERUTIL_H
#define MANTARAY_BACKEND_CONTAINERUTIL_H

#include <algorithm>
#include <array>

#include <MantaRay/Common/Alignment.h>
#include <MantaRay/Common/Integral.h>

template<typename T, MantaRay::s00 N>
struct alignas(MantaRay::Alignment) Array
{

    constexpr static MantaRay::s00 GuardSize = MantaRay::Alignment / sizeof(T);

    constexpr static T Sentinel = static_cast<T>(0x5A5A5A5A);

    std::array<T, N        > Values {};
    std::array<T, GuardSize> Guard  {};

    constexpr Array() { Guard.fill(Sentinel); }

          T* data()       { return Values.data(); }
    const T* data() const { return Values.data(); }

          T& operator [](const MantaRay::s00 i)       { return Values[i]; }
    const T& operator [](const MantaRay::s00 i) const { return Values[i]; }

    [[nodiscard]]
    bool Intact() const { return std::ranges::all_of(Guard, [](const T value) { return value == Sentinel; }); }

};

template<typename T, MantaRay::s00 N, T A>
constexpr Array<T, N> Generate()
{
    Array<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++) result[i] = static_cast<T>(i % 2 == 0 ? A : 0);

    return result;
}

template<typename T, MantaRay::s00 N, T A>
constexpr Array<T, N> Ramp()
{
    Array<T, N> result;

    for (MantaRay::s00 i = 0; i < N; i++) result[i] = static_cast<T>(A + static_cast<T>(i % 128));

    return result;
}

#endif //MANTARAY_BACKEND_CONTAINERUTIL_H
