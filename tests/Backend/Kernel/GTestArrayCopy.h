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

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i16_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i32_512)
{
    constexpr MantaRay::s00 N = 512;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i08_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i08, N> a = Generate<MantaRay::i08, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i08, N> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i16_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i16, N> a = Generate<MantaRay::i16, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i16, N> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i32_2048)
{
    constexpr MantaRay::s00 N = 2048;

    ALIGN MantaRay::Array<MantaRay::i32, N> a = Generate<MantaRay::i32, N, 30>();
    ALIGN MantaRay::Array<MantaRay::i32, N> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);
}

TEST(ArrayCopy, i16_2x512)
{
    constexpr MantaRay::s00 N = 2;
    constexpr MantaRay::s00 M = 512;

    ALIGN MantaRay::NArray<MantaRay::i16, N, M> a = Generate<MantaRay::i16, N, M, 30>();
    ALIGN MantaRay::NArray<MantaRay::i16, N, M> b;

    MantaRay::Backend::Kernel::Copy(a, b);

    for (MantaRay::s00 i = 0; i < N; i++) for (MantaRay::s00 j = 0; j < M; j++) EXPECT_EQ(a[i][j], b[i][j]);
}

#endif //MANTARAY_GTESTARRAYCOPY_H
