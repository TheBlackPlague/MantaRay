//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_GTESTDIFFERENTIAL_H
#define MANTARAY_BACKEND_GTESTDIFFERENTIAL_H

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Backend/Target.h>
#include <MantaRay/Backend/ISA/Scalar.h>
#include <MantaRay/Common/Interval.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/Delta.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "../Runtime/NetworkUtil.h"

using namespace MantaRay;

namespace DifferentialTest
{

    using Scalar = Backend::ISA::Scalar;

    constexpr bool NativeIsScalar = std::is_same_v<Backend::Native, Scalar>;

    struct Fills
    {

        Fill Transformer;
        Fill Dense      ;

    };

    constexpr std::array<Fills, 9> Combinations {{
        { Fill::Maximum , Fill::Maximum  }, { Fill::Minimum , Fill::Maximum  },
        { Fill::Maximum , Fill::Minimum  }, { Fill::Maximum , Fill::Extremes },
        { Fill::Minimum , Fill::Extremes }, { Fill::Extremes, Fill::Extremes },
        { Fill::Random  , Fill::Extremes }, { Fill::Random  , Fill::Random   },
        { Fill::Minimum , Fill::Minimum  }
    }};

    template<typename A>
    struct Pair
    {

        std::unique_ptr<Network<A, Scalar>> Reference = std::make_unique<Network<A, Scalar>>();
        std::unique_ptr<Network<A        >> Native    = std::make_unique<Network<A        >>();

        Pair(std::mt19937_64& random, const Fills fills)
        {
            std::vector<std::byte> bytes = RandomBytes<A>(random, fills.Transformer, fills.Dense);

            if (random() % 2 == 0) Pad(bytes);

            EXPECT_TRUE(static_cast<bool>(Reference->Load(bytes)));
            EXPECT_TRUE(static_cast<bool>(Native   ->Load(bytes)));
        }

    };

    inline std::vector<u16> RandomPieces(std::mt19937_64& random, const s00 count, const bool kings)
    {
        std::vector<u16> pieces;

        u64 occupied = 0;

        const auto place = [&](const u08 piece, const u08 color) {
            u08 square;

            do square = static_cast<u08>(random() % 64); while (occupied >> square & 1);

            occupied |= u64 { 1 } << square;

            pieces.push_back(PieceCode(piece, color, square));
        };

        if (kings) for (u08 color = 0; color < 2; color++) place(KingPiece, color);

        const u08 kinds = kings ? 5 : 6;

        while (pieces.size() < count) place(static_cast<u08>(random() % kinds), static_cast<u08>(random() % 2));

        return pieces;
    }

    template<typename A>
    void CheckNetworks(std::mt19937_64& random, const s00 positions)
    {
        for (const Fills fills : Combinations) {
            const Pair<A> pair(random, fills);

            for (s00 position = 0; position < positions; position++) {
                const s00 count = position % 4 == 0 ? 32 : 2 + random() % 31;

                const std::vector<u16> pieces = RandomPieces(random, count, NeedsKings<A>);

                for (u08 stm = 0; stm < 2; stm++) {
                    const i32 expected = pair.Reference->Evaluate(pieces, stm);
                    const i32 actual   = pair.Native   ->Evaluate(pieces, stm);

                    ASSERT_EQ(actual, expected) << Network<A>::Describe() << ", fills "
                                                << s00 { static_cast<u08>(fills.Transformer) } << "/"
                                                << s00 { static_cast<u08>(fills.Dense      ) } << ", "
                                                << count << " pieces, stm " << s00 { stm };
                }
            }
        }
    }

    template<typename A, UpdatePolicy Policy>
    void WalkStacks(std::mt19937_64& random, const Fills fills, const s00 operations)
    {
        constexpr s00 H = Network<A>::HiddenSize;

        const Pair<A> pair(random, fills);

        const auto reference = std::make_unique<Stack<A, Policy, 33, Scalar>>(*pair.Reference);
        const auto native    = std::make_unique<Stack<A, Policy             >>(*pair.Native   );

        Walk walk(random, NeedsKings<A>);

        walk.Reset(StartPosition());

        ResetFromPieces(*reference, walk.Pieces());
        ResetFromPieces(*native   , walk.Pieces());

        if (fills.Transformer == Fill::Maximum || fills.Transformer == Fill::Minimum) {
            const Interval range   = A::Signals[0].Range;
            const i64      extreme = fills.Transformer == Fill::Maximum ? range.Max : range.Min;

            for (u08 perspective = 0; perspective < 2; perspective++)
            for (s00 i           = 0; i < H; i++) ASSERT_EQ(native->Values(perspective)[i], extreme);
        }

        s00 evaluations = 0;

        for (s00 operation = 0; operation < operations; operation++) {
            const u64 roll = random() % 100;

            if (roll < 45 && walk.Depth() < 32) {
                const Delta delta = walk.Next();

                reference->Push(delta);
                native   ->Push(delta);
            } else if (roll < 75 && walk.Depth() > 0) {
                walk.Pop();

                reference->Pop();
                native   ->Pop();
            } else if (roll < 77) {
                walk.Reset(walk.RandomPosition());

                ResetFromPieces(*reference, walk.Pieces());
                ResetFromPieces(*native   , walk.Pieces());
            } else {
                walk.Evaluate();

                const auto stm = static_cast<u08>(random() % 2);

                const i32 expected = EvaluateWithBoards(*reference, walk.Pieces(), stm);
                const i32 actual   = EvaluateWithBoards(*native   , walk.Pieces(), stm);

                ASSERT_EQ(expected, pair.Reference->Evaluate(walk.Pieces(), stm)) << "operation " << operation;
                ASSERT_EQ(actual  , expected                                    ) << "operation " << operation;

                for (u08 perspective = 0; perspective < 2; perspective++) {
                    const std::span<const typename Network<A>::Accumulator> values(native->Values(perspective), H);

                    ASSERT_TRUE(SameValues(*reference, values, perspective)) << "operation " << operation;
                }

                evaluations++;
            }
        }

        EXPECT_GT(evaluations, operations / 5);
    }

    template<typename A>
    void CheckStacks(std::mt19937_64& random, const s00 operations)
    {
        for (const Fills fills : { Combinations[0], Combinations[1], Combinations[5], Combinations[7] }) {
            WalkStacks<A, UpdatePolicy::Lazy >(random, fills, operations);
            WalkStacks<A, UpdatePolicy::Eager>(random, fills, operations);
        }
    }

}

TEST(Differential, ScalarVsNative)
{
    if (DifferentialTest::NativeIsScalar) GTEST_SKIP() << "the native target is the scalar reference";

    std::mt19937_64 random(0x4449464645524E54);

    DifferentialTest::CheckNetworks<Examples::Starshard        >(random, 32);
    DifferentialTest::CheckNetworks<Examples::Aurora           >(random, 32);
    DifferentialTest::CheckNetworks<Examples::Flare            >(random, 32);
    DifferentialTest::CheckNetworks<Examples::SimpleSCReLU     >(random, 64);
    DifferentialTest::CheckNetworks<Examples::WideSCReLU       >(random, 32);
    DifferentialTest::CheckNetworks<Examples::OutputBucketed   >(random, 64);
    DifferentialTest::CheckNetworks<Examples::KingBucketed     >(random, 64);
    DifferentialTest::CheckNetworks<Examples::MultiLayer       >(random, 64);
    DifferentialTest::CheckNetworks<Examples::WideAccumulator  >(random, 64);
    DifferentialTest::CheckNetworks<Examples::SquareFirstSCReLU>(random, 64);
    DifferentialTest::CheckNetworks<Examples::WidenedSCReLU    >(random, 64);
    DifferentialTest::CheckNetworks<Examples::LinearReLU       >(random, 64);
    DifferentialTest::CheckNetworks<Examples::WidenedSqrReLU   >(random, 64);
    DifferentialTest::CheckNetworks<Examples::PackedPairwise   >(random, 64);
    DifferentialTest::CheckNetworks<Examples::ShiftedPairwise  >(random, 64);
}

TEST(Differential, AccumulatorStackWalks)
{
    if (DifferentialTest::NativeIsScalar) GTEST_SKIP() << "the native target is the scalar reference";

    std::mt19937_64 random(0x5354414B57414C4B);

    DifferentialTest::CheckStacks<Examples::Flare            >(random,  500);
    DifferentialTest::CheckStacks<Examples::SimpleSCReLU     >(random, 1000);
    DifferentialTest::CheckStacks<Examples::KingBucketed     >(random, 1000);
    DifferentialTest::CheckStacks<Examples::MultiLayer       >(random, 1000);
    DifferentialTest::CheckStacks<Examples::WideAccumulator  >(random, 1000);
    DifferentialTest::CheckStacks<Examples::SquareFirstSCReLU>(random, 2000);
    DifferentialTest::CheckStacks<Examples::WidenedSCReLU    >(random, 2000);
    DifferentialTest::CheckStacks<Examples::LinearReLU       >(random, 2000);
    DifferentialTest::CheckStacks<Examples::WidenedSqrReLU   >(random, 2000);
    DifferentialTest::CheckStacks<Examples::PackedPairwise   >(random, 2000);
    DifferentialTest::CheckStacks<Examples::ShiftedPairwise  >(random, 2000);
}

#endif //MANTARAY_BACKEND_GTESTDIFFERENTIAL_H
