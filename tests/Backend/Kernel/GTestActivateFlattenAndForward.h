//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_GTESTACTIVATEFLATTENANDFORWARD_H
#define MANTARAY_GTESTACTIVATEFLATTENANDFORWARD_H

#include <gtest/gtest.h>

#include <MantaRay/Backend/Kernel/ActivateFlattenAndForward.h>

#include "../ContainerUtil.h"

// ReSharper disable CppLocalVariableMayBeConst
TEST(ActivateFlattenAndForward, i16_512)
{
    constexpr MantaRay::usize N = 512;
    constexpr MantaRay::usize M = 1;

    using ClippedReLU = MantaRay::ClippedReLU<MantaRay::i16, 0, 255>;

    HWY_ALIGN MantaRay::Array<MantaRay::i16, N> x0 = Generate<MantaRay::i16, N, 30>();
    HWY_ALIGN MantaRay::Array<MantaRay::i16, N> x1 = Generate<MantaRay::i16, N, 04>();

    HWY_ALIGN MantaRay::Array<MantaRay::i16, N * 2 * M> w = Generate<MantaRay::i16, N * 2 * M, 0>();

    HWY_ALIGN MantaRay::Array<MantaRay::i16, M> b = Generate<MantaRay::i16, M, 0>();

    HWY_ALIGN auto result = MantaRay::ActivateFlattenAndForward<ClippedReLU, MantaRay::i16, MantaRay::i32, N, M>(x0, x1, w, b);
    EXPECT_EQ(result[0], 0);
}

TEST(ActivateFlattenAndForward, i16_2048)
{
    constexpr MantaRay::usize N = 2048;
    constexpr MantaRay::usize M = 1;

    using ClippedReLU = MantaRay::ClippedReLU<MantaRay::i16, 0, 255>;

    HWY_ALIGN MantaRay::Array<MantaRay::i16, N> x0 = Generate<MantaRay::i16, N, 30>();
    HWY_ALIGN MantaRay::Array<MantaRay::i16, N> x1 = Generate<MantaRay::i16, N, 04>();

    HWY_ALIGN MantaRay::Array<MantaRay::i16, N * 2 * M> w = Generate<MantaRay::i16, N * 2 * M, 0>();

    HWY_ALIGN MantaRay::Array<MantaRay::i16, M> b = Generate<MantaRay::i16, M, 0>();

    HWY_ALIGN auto result = MantaRay::ActivateFlattenAndForward<ClippedReLU, MantaRay::i16, MantaRay::i32, N, M>(x0, x1, w, b);
    EXPECT_EQ(result[0], 0);
}

// ReSharper restore CppLocalVariableMayBeConst

#endif //MANTARAY_GTESTACTIVATEFLATTENANDFORWARD_H
