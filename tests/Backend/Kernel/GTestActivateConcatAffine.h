//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_GTESTACTIVATECONCATAFFINE_H
#define MANTARAY_GTESTACTIVATECONCATAFFINE_H

#include <limits>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/Quantization.h>
#include <MantaRay/Backend/Kernel/Fused/ActivateConcatAffine.h>

template<MantaRay::s00 N, MantaRay::s00 M>
void CheckActivateConcatAffine()
{
    using namespace MantaRay;

    using Q = Quantization<255, 64, 400>;
    using A = ClippedReLU<0, 1>;

    ALIGN  Array<i16,        N> first   {};
    ALIGN  Array<i16,        N> second  {};
    ALIGN NArray<i16, M, 2 * N> weights {};
    ALIGN  Array<i32, M       > bias    {};

    Array<i32, M> expected {};

    for (s00 j = 0; j < N; ++j) {
        first [j] = static_cast<i16>(      static_cast<i32>(j % 400) - 50);
        second[j] = static_cast<i16>(300 - static_cast<i32>(j % 400)     );
    }

    for (s00 i = 0; i < M; ++i) {
        expected[i] = bias[i] = 40000 + static_cast<i32>(i);

        for (s00 j = 0; j < N; ++j) {
            weights[i][    j] = static_cast<i16>(static_cast<i32>((i + j) % 11) - 5);
            weights[i][N + j] = static_cast<i16>(static_cast<i32>((i + j) %  7) - 3);

            const i32 a = std::clamp<i32>(first [j], 0, 255);
            const i32 b = std::clamp<i32>(second[j], 0, 255);

            expected[i] += a * weights[i][j] + b * weights[i][N + j];
        }
    }

    const auto result = Backend::Kernel::ActivateConcatAffine<A, Q, N, M>(first, second, weights, bias);

    for (s00 i = 0; i < M; ++i) EXPECT_EQ(result[i], expected[i]);
}

TEST(ActivateConcatAffine, i16_512 ) { CheckActivateConcatAffine< 512, 1>(); }
TEST(ActivateConcatAffine, i16_2048) { CheckActivateConcatAffine<2048, 4>(); }
TEST(ActivateConcatAffine, i16_384 ) { CheckActivateConcatAffine< 384, 1>(); }
TEST(ActivateConcatAffine, i16_8   ) { CheckActivateConcatAffine<   8, 2>(); }
TEST(ActivateConcatAffine, i16_24  ) { CheckActivateConcatAffine<  24, 2>(); }
TEST(ActivateConcatAffine, i16_17  ) { CheckActivateConcatAffine<  17, 2>(); }

template<MantaRay::s00 Lanes>
void CheckAccumulateDot()
{
    using namespace MantaRay;
    namespace Native = Backend::Native;

    constexpr std::array<i16, 8> values { -32768, 32767, -1, 0, 1, -12345, 23456, -32768 };
    Array<i16, Lanes> left {}, right {};
    Array<i32, Lanes / 2> initial {};
    u32 expected = 0;

    for (s00 i = 0; i < initial.size(); ++i) {
        initial[i] = i % 2 == 0 ? std::numeric_limits<i32>::max() : std::numeric_limits<i32>::min();
        expected += static_cast<u32>(initial[i]);
    }

    auto sum = Native::Load<i32, Lanes / 2>(initial.data());
    for (s00 iteration = 0; iteration < 17; ++iteration) {
        for (s00 i = 0; i < Lanes; ++i) {
            // Include repeated INT16_MIN products, mixed signs, and wrapping
            // accumulated sums; the reference never performs signed overflow.
            left [i] = iteration == 0 ? -32768 : values[(i + iteration) % values.size()];
            right[i] = iteration == 0 ? -32768 : values[(3 * i + iteration) % values.size()];
            expected += static_cast<u32>(static_cast<i32>(left[i]) * right[i]);
        }

        sum = Native::AccumulateDot(sum, Native::Load<i16, Lanes>(left.data()), Native::Load<i16, Lanes>(right.data()));
        EXPECT_EQ(static_cast<u32>(Native::Sum(sum)), expected);
    }
}

TEST(AccumulateDot, TwoLanes      ) { CheckAccumulateDot< 2>(); }
TEST(AccumulateDot, EightLanes    ) { CheckAccumulateDot< 8>(); }
TEST(AccumulateDot, SixteenLanes  ) { CheckAccumulateDot<16>(); }
TEST(AccumulateDot, ThirtyTwoLanes) { CheckAccumulateDot<32>(); }

TEST(ActivateConcatAffine, FullRangeIdentityWrapping)
{
    using namespace MantaRay;
    using Q = Quantization<1, 1, 1>;
    constexpr s00 N = 64;

    Array<i16, N> first {}, second {};
    NArray<i16, 2, 2 * N> weights {};
    Array<i32, 2> bias { std::numeric_limits<i32>::max(), std::numeric_limits<i32>::min() };
    std::array<u32, 2> expected { static_cast<u32>(bias[0]), static_cast<u32>(bias[1]) };

    for (s00 i = 0; i < N; ++i) {
        first[i] = i % 3 == 0 ? -32768 : 32767;
        second[i] = i % 5 == 0 ? 12345 : -32768;
        for (s00 output = 0; output < 2; ++output) {
            weights[output][i] = output == 0 ? -32768 : 32767;
            weights[output][N + i] = i % 2 == 0 ? -32768 : -12345;
            expected[output] += static_cast<u32>(static_cast<i32>(first[i]) * weights[output][i]);
            expected[output] += static_cast<u32>(static_cast<i32>(second[i]) * weights[output][N + i]);
        }
    }

    const auto result = Backend::Kernel::ActivateConcatAffine<Identity, Q, N, 2>(first, second, weights, bias);
    for (s00 output = 0; output < 2; ++output) EXPECT_EQ(static_cast<u32>(result[output]), expected[output]);
}

#endif
