//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_GTESTARRAYADD_H
#define MANTARAY_GTESTARRAYADD_H

#include <gtest/gtest.h>

#include <MantaRay/Backend/Kernel/ArrayAdd.h>

#include "../ContainerUtil.h"

// ReSharper disable CppLocalVariableMayBeConst
TEST(ArrayAdd, i08_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i08, N> a = Generate<MantaRay::i08, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i08, N> b = Generate<MantaRay::i08, N, 04>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], static_cast<MantaRay::i08>(i % 2 == 0 ? 30 + 04 : 0));
}

TEST(ArrayAdd, i16_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b = Generate<MantaRay::i16, N, 04>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], static_cast<MantaRay::i16>(i % 2 == 0 ? 30 + 04 : 0));
}

TEST(ArrayAdd, i32_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b = Generate<MantaRay::i32, N, 04>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], i % 2 == 0 ? 30 + 04 : 0);
}

TEST(ArrayAdd, i08_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i08, N> a = Generate<MantaRay::i08, N, -30>();
    ALIGN MantaRay::Array<MantaRay::i08, N> b = Generate<MantaRay::i08, N,  54>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], static_cast<MantaRay::i08>(i % 2 == 0 ? -30 + 54 : 0));
}

TEST(ArrayAdd, i16_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, -30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b = Generate<MantaRay::i16, N,  54>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], static_cast<MantaRay::i16>(i % 2 == 0 ? -30 + 54 : 0));
}

TEST(ArrayAdd, i32_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, -30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b = Generate<MantaRay::i32, N,  54>();

    MantaRay::ArrayAdd(a, b);

    for (MantaRay::s00 i = 0; i < N; i++)
        EXPECT_EQ(a[i], i % 2 == 0 ? -30 + 54 : 0);
}

// ReSharper restore CppLocalVariableMayBeConst

#endif //MANTARAY_GTESTARRAYADD_H
