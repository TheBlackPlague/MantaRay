//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_GTESTARRAYCOPY_H
#define MANTARAY_GTESTARRAYCOPY_H

#include <gtest/gtest.h>

#include <MantaRay/Backend/Kernel/ArrayCopy.h>

#include "../ContainerUtil.h"

// ReSharper disable CppLocalVariableMayBeConst
TEST(ArrayCopy, i08_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i08, N> a = Generate<MantaRay::i08, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i08, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i16_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i32_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i08_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i08, N> a = Generate<MantaRay::i08, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i08, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i16_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i32_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b;

    MantaRay::ArrayCopy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], b[i]);
}

#endif //MANTARAY_GTESTARRAYCOPY_H
