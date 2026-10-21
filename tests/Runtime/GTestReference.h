//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTREFERENCE_H
#define MANTARAY_RUNTIME_GTESTREFERENCE_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Architecture/FeatureSet/Chess768.h>
#include <MantaRay/Architecture/FeatureSet/KingBuckets.h>
#include <MantaRay/Common/List.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

namespace ReferenceTest
{

    struct Activation
    {

        bool Known   = false;
        bool Squares = false;
        i64  Low     = 0;
        i64  High    = std::numeric_limits<i32>::max();

    };

    template<typename L>
    constexpr inline Activation ActivationOf {};

    template<i32 Min, i32 Max>
    constexpr inline Activation ActivationOf<ClippedReLU<Min, Max>> { true, false, Min, Max };

    template<i32 Min, i32 Max>
    constexpr inline Activation ActivationOf<SquaredClippedReLU<Min, Max>> { true, true, Min, Max };

    template<>
    constexpr inline Activation ActivationOf<ReLU> { true, false };

    template<>
    constexpr inline Activation ActivationOf<SquaredReLU> { true, true };

    inline u32 Plain(const u08 perspective, const u16 code, const u32 flip)
    {
        const u32 square = SquareOf(code) ^ 56 * perspective ^ flip;

        return 384 * (ColorOf(code) != perspective) + 64 * PieceOf(code) + square;
    }

    template<typename F>
    struct Features;

    template<>
    struct Features<Chess768>
    {

        static u32 Index(const u08 perspective, const u08, const u16 code) { return Plain(perspective, code, 0); }

    };

    template<std::array<u08, 32> Layout>
    struct Features<MirroredKingBuckets<Layout>>
    {

        static u32 Index(const u08 perspective, const u08 king, const u16 code)
        {
            const u32 square = king ^ 56 * perspective;
            const u32 flip   = square % 8 > 3 ? 7 : 0;

            return 768 * Layout[square / 8 * 4 + (square % 8 ^ flip)] + Plain(perspective, code, flip);
        }

    };

    struct Values
    {

        std::vector<std::vector<i64>> Sides       {};
        std::vector<double>           Real        {};
        bool                          Float       = false;
        i64                           Numerator   = 1;
        i64                           Denominator = 1;

    };

    struct Result
    {

        i32  Value = 0;
        bool Float = false;

    };

    template<typename A>
    class Reference
    {

        using T = A::Transformer;
        using F = T::Features;

        constexpr static s00 H = T::OutputSize;

        std::vector<std::vector<double>> Tensors;

        template<typename S>
        void Read(const std::span<const std::byte> bytes, s00& position, const s00 count)
        {
            using V = S::Type;
            using U = std::conditional_t<sizeof(V) == 1, u08, std::conditional_t<sizeof(V) == 2, u16, u32>>;

            for (double& value : Tensors.emplace_back(count)) {
                U bits = 0;

                for (s00 i = 0; i < sizeof(V); i++)
                    bits |= static_cast<U>(std::to_integer<U>(bytes[position++]) << 8 * i);

                value = std::bit_cast<V>(bits);
            }
        }

        template<typename W>
        static double Weight(const std::vector<double>& tensor, const s00 in, const s00 out, const s00 i, const s00 o)
        {
            return tensor[W::IsTransposed ? o * in + i : i * out + o];
        }

        static i64 Rectify(const Activation activation, const i64 value, const Values& values)
        {
            const i64 low  = activation.Low  * values.Numerator / values.Denominator;
            const i64 high = activation.High * values.Numerator / values.Denominator;
            const i64 x    = std::clamp(value, low, high);

            return activation.Squares ? x * x : x;
        }

        template<typename L>
        void Apply(Values& values, s00& tensor, const s00 pieces, i32& output) const
        {
            if constexpr (ActivationOf<L>.Known) {
                constexpr Activation R = ActivationOf<L>;

                for (double& x : values.Real) {
                    x = std::clamp<double>(x, R.Low, R.High);

                    if (R.Squares) x *= x;
                }

                for (std::vector<i64>& side : values.Sides)
                for (i64&              x    : side        ) x = Rectify(R, x, values);

                if (R.Squares) {
                    values.Numerator   *= values.Numerator  ;
                    values.Denominator *= values.Denominator;
                }
            } else if constexpr (requires { typename L::Activation; }) {
                constexpr Activation R = ActivationOf<typename L::Activation>;

                for (std::vector<i64>& side : values.Sides) {
                    const s00 half = side.size() / 2;

                    std::vector<i64> products(half);

                    for (s00 j = 0; j < half; j++)
                        products[j] = Rectify(R, side[j], values) * Rectify(R, side[half + j], values) >> L::Shift;

                    side = std::move(products);
                }

                const i64 numerator   = R.Squares ? values.Numerator   * values.Numerator   : values.Numerator  ;
                const i64 denominator = R.Squares ? values.Denominator * values.Denominator : values.Denominator;

                values.Numerator   = numerator   * numerator;
                values.Denominator = denominator * denominator << L::Shift;
            } else if constexpr (std::is_same_v<L, Concat>) {
                values.Sides[0].insert(values.Sides[0].end(), values.Sides[1].begin(), values.Sides[1].end());
                values.Sides.pop_back();
            } else if constexpr (requires { typename L::Buckets; }) {
                constexpr s00 N = L::Buckets::Count;

                const s00 bucket = (pieces - 2) / ((32 + N - 1) / N);

                const auto select = [&]<typename V>(std::vector<V>& items) {
                    const s00 size = items.size() / N;

                    items = std::vector<V>(items.begin() + bucket * size, items.begin() + (bucket + 1) * size);
                };

                if (values.Float) select(values.Real    );
                else              select(values.Sides[0]);
            } else if constexpr (requires { typename L::WeightSpec; }) {
                using W = L::WeightSpec;
                using S = L::Schedule;

                const std::vector<double>& weights = Tensors[tensor++];
                const std::vector<double>& biases  = Tensors[tensor++];

                const bool   real  = values.Float || std::is_same_v<S, Dequantize>;
                const double scale = static_cast<double>(values.Denominator) / (values.Numerator * W::Quantization);

                std::vector<i64   > integers;
                std::vector<double> reals   ;

                for (s00 o = 0; o < L::OutputSize; o++) {
                    i64    sum   = 0;
                    double total = 0;

                    for (s00 i = 0; i < L::InputSize; i++) {
                        const double weight = Weight<W>(weights, L::InputSize, L::OutputSize, i, o);

                        if (values.Float) total += values.Real[i] * weight;
                        else              sum   += values.Sides[0][i] * static_cast<i64>(weight);
                    }

                    if constexpr (requires { S::Divisor; }) sum /= S::Divisor;
                    if constexpr (requires { S::Shift;   }) sum >>= S::Shift;

                    if (real) reals   .push_back((values.Float ? total : static_cast<double>(sum) * scale) + biases[o]);
                    else      integers.push_back(sum + static_cast<i64>(biases[o]));
                }

                if (real) values = { .Real  = std::move(reals), .Float = true };
                else      values = { .Sides = { std::move(integers) }, .Numerator = L::BiasSpec::Quantization };
            } else if constexpr (requires { L::OutputScale; }) {
                const i64 scale = L::OutputScale;

                if (values.Float) output = static_cast<i32>(values.Real[0] * static_cast<double>(scale));
                else              output = static_cast<i32>(values.Sides[0][0] * scale * values.Denominator /
                                                            values.Numerator);
            } else static_assert(false, "the reference has no semantics for this layer");
        }

        public:
        explicit Reference(const std::span<const std::byte> bytes)
        {
            s00 position = 0;

            Read<typename T::WeightSpec>(bytes, position, F::InputSize * H);
            Read<typename T::BiasSpec  >(bytes, position,                H);

            [&]<typename... L>(List<L...>) {
                ([&] {
                    if constexpr (requires { typename L::WeightSpec; }) {
                        Read<typename L::WeightSpec>(bytes, position, L::InputSize * L::OutputSize);
                        Read<typename L::BiasSpec  >(bytes, position,                L::OutputSize);
                    }
                }(), ...);
            }(typename A::Layers {});

            EXPECT_EQ(position, bytes.size());
        }

        Result Evaluate(const std::vector<u16>& pieces, const u08 stm) const
        {
            using W = T::WeightSpec;

            std::array<u08, 2> kings {};

            for (const u16 code : pieces) if (PieceOf(code) == KingPiece) kings[ColorOf(code)] = SquareOf(code);

            Values values { .Numerator = W::Quantization };

            for (const u08 perspective : { stm, static_cast<u08>(stm ^ 1) }) {
                std::vector<i64>& side = values.Sides.emplace_back(Tensors[1].begin(), Tensors[1].end());

                for (const u16 code : pieces) {
                    const u32 feature = Features<F>::Index(perspective, kings[perspective], code);

                    for (s00 j = 0; j < H; j++)
                        side[j] += static_cast<i64>(Weight<W>(Tensors[0], F::InputSize, H, feature, j));
                }
            }

            s00 tensor = 2;
            i32 output = 0;

            [&]<typename... L>(List<L...>) {
                (Apply<L>(values, tensor, pieces.size(), output), ...);
            }(typename A::Layers {});

            return { output, values.Float };
        }

    };

    template<typename A>
    void Check(std::mt19937_64& random, const s00 positions)
    {
        for (const Fill fill : { Fill::Random, Fill::Extremes }) {
            const std::vector<std::byte> bytes = RandomBytes<A>(random, fill, fill);

            const Reference<A> reference(bytes);

            const auto network = std::make_unique<Network<A>>();

            ASSERT_TRUE(network->Load(bytes));

            const auto stack = std::make_unique<Stack<A, UpdatePolicy::Lazy, 1>>(*network);

            Walk walk(random, NeedsKings<A>);

            for (s00 position = 0; position < positions; position++) {
                const std::vector<u16> pieces = position == 0 ? StartPosition() : walk.RandomPosition();

                ResetFromPieces(*stack, pieces);

                for (u08 stm = 0; stm < 2; stm++) {
                    const Result expected = reference.Evaluate(pieces, stm);
                    const i64    limit    = expected.Float ? 1 : 0;

                    const i32 direct  = network->Evaluate(pieces, stm);
                    const i32 stacked = EvaluateWithBoards(*stack, pieces, stm);

                    ASSERT_LE(std::abs(i64 { direct  } - expected.Value), limit)
                        << Network<A>::Describe() << ", fill " << s00 { static_cast<u08>(fill) } << ", "
                        << pieces.size() << " pieces, stm " << s00 { stm } << ": network " << direct << ", reference "
                        << expected.Value;

                    ASSERT_LE(std::abs(i64 { stacked } - expected.Value), limit)
                        << Network<A>::Describe() << ", fill " << s00 { static_cast<u08>(fill) } << ", "
                        << pieces.size() << " pieces, stm " << s00 { stm } << ": stack " << stacked << ", reference "
                        << expected.Value;
                }
            }
        }
    }

}

TEST(Reference, Perspective)
{
    std::mt19937_64 random(0x5245465045525350);

    ReferenceTest::Check<Examples::Starshard >(random, 32);
    ReferenceTest::Check<Examples::Aurora    >(random, 32);
    ReferenceTest::Check<Examples::Flare     >(random, 32);
    ReferenceTest::Check<Examples::LinearReLU>(random, 64);
}

TEST(Reference, Squared)
{
    std::mt19937_64 random(0x5245465351554152);

    ReferenceTest::Check<Examples::SimpleSCReLU     >(random, 64);
    ReferenceTest::Check<Examples::WideSCReLU       >(random, 32);
    ReferenceTest::Check<Examples::SquareFirstSCReLU>(random, 64);
    ReferenceTest::Check<Examples::WidenedSCReLU    >(random, 64);
    ReferenceTest::Check<Examples::WidenedSqrReLU   >(random, 64);
    ReferenceTest::Check<Examples::WideAccumulator  >(random, 64);
}

TEST(Reference, Bucketed)
{
    std::mt19937_64 random(0x5245464255434B54);

    ReferenceTest::Check<Examples::OutputBucketed>(random, 64);
    ReferenceTest::Check<Examples::KingBucketed  >(random, 64);
}

TEST(Reference, Pairwise)
{
    std::mt19937_64 random(0x5245465041495257);

    ReferenceTest::Check<Examples::MultiLayer     >(random, 64);
    ReferenceTest::Check<Examples::PackedPairwise >(random, 64);
    ReferenceTest::Check<Examples::ShiftedPairwise>(random, 64);
}

#endif //MANTARAY_RUNTIME_GTESTREFERENCE_H
