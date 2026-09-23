#pragma once

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
