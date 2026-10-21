//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTFINNYTABLE_H
#define MANTARAY_RUNTIME_GTESTFINNYTABLE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Common/Alignment.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/FinnyTable.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

namespace FinnyTableTest
{

    template<typename N>
    constexpr bool HasGeneration = requires (const N& network) { network.Generation(); };

    template<typename A, UpdatePolicy Policy>
    void ExpectFresh(Stack<A, Policy>& stack, const Network<A>& network, const std::vector<u16>& pieces)
    {
        constexpr s00 H = Network<A>::HiddenSize;

        const auto fresh = std::make_unique<Stack<A, Policy>>(network);

        ResetFromPieces(stack , pieces);
        ResetFromPieces(*fresh, pieces);

        for (u08 perspective = 0; perspective < 2; perspective++) {
            const std::span<const typename Network<A>::Accumulator> values(fresh->Values(perspective), H);

            ASSERT_TRUE(SameValues(stack, values, perspective));
        }

        for (u08 stm = 0; stm < 2; stm++)
            ASSERT_EQ(EvaluateWithBoards(stack, pieces, stm), network.Evaluate(pieces, stm));
    }

    template<typename A, UpdatePolicy Policy>
    std::vector<std::vector<u16>> Visit(std::mt19937_64& random, Stack<A, Policy>& stack, const s00 operations)
    {
        Walk walk(random, true);

        walk.Reset(StartPosition());

        ResetFromPieces(stack, walk.Pieces());

        std::vector<std::vector<u16>> visited { walk.Pieces() };

        for (s00 operation = 0; operation < operations; operation++) {
            if (walk.Depth() < 32 && random() % 3 != 0) {
                stack.Push(walk.Next());

                (void) EvaluateWithBoards(stack, walk.Pieces(), static_cast<u08>(random() % 2));

                visited.push_back(walk.Pieces());
            } else if (walk.Depth() > 0) {
                walk.Pop();

                stack.Pop();
            }
        }

        return visited;
    }

    template<typename A, UpdatePolicy Policy>
    void CheckReload()
    {
        std::mt19937_64 random(0x52454C4F41444544);

        const std::vector<std::byte> first  = RandomBytes<A>(random);
        const std::vector<std::byte> second = RandomBytes<A>(random);

        const auto network = std::make_unique<Network<A>>();

        ASSERT_TRUE(network->Load(first));

        const auto stack = std::make_unique<Stack<A, Policy>>(*network);

        const std::vector<std::vector<u16>> visited = Visit(random, *stack, 400);

        ASSERT_TRUE(network->Load(second));

        for (s00 i = 0; i < visited.size(); i++) {
            ExpectFresh(*stack, *network, visited[i]);

            ASSERT_FALSE(testing::Test::HasFatalFailure()) << "position " << i;
        }
    }

    template<UpdatePolicy Policy>
    void CheckAssignment()
    {
        using A = Examples::KingBucketed;

        std::mt19937_64 random(0x41535349474E4544);

        const auto first  = RandomNetwork<A>(random);
        const auto second = RandomNetwork<A>(random);

        const auto stack = std::make_unique<Stack<A, Policy>>(*first);

        const std::vector<std::vector<u16>> visited = Visit(random, *stack, 200);

        *first = *second;

        for (s00 i = 0; i < visited.size(); i++) {
            ExpectFresh(*stack, *first, visited[i]);

            ASSERT_FALSE(testing::Test::HasFatalFailure()) << "position " << i;
        }
    }

    template<UpdatePolicy Policy>
    void CheckStackBeforeFirstLoad()
    {
        using A = Examples::KingBucketed;

        std::mt19937_64 random(0x46495253544C4F44);

        const auto network = std::make_unique<Network<A>>();
        const auto stack   = std::make_unique<Stack<A, Policy>>(*network);

        ASSERT_TRUE(network->Load(RandomBytes<A>(random)));

        ExpectFresh(*stack, *network, StartPosition());

        Walk walk(random, true);

        walk.Reset(StartPosition());

        for (s00 i = 0; i < 16; i++) {
            stack->Push(walk.Next());

            for (u08 stm = 0; stm < 2; stm++)
                ASSERT_EQ(EvaluateWithBoards(*stack, walk.Pieces(), stm), network->Evaluate(walk.Pieces(), stm));
        }
    }

}

TEST(FinnyTable, EntryEqualsFreshRefresh)
{
    using A = Examples::KingBucketed;
    using F = A::Transformer::Features;
    using T = Network<A>::Accumulator;

    constexpr s00 H = Network<A>::HiddenSize;

    std::mt19937_64 random(0x46494E4E59544142);

    const auto network = RandomNetwork<A>(random);
    const auto table   = std::make_unique<FinnyTable<A>>();

    table->Clear(*network);

    const auto fresh = [&](const u08 perspective, const u08 key, const std::vector<u16>& pieces) {
        std::array<T, H> values;

        std::copy_n(network->FeatureBias(), H, values.begin());

        for (const u16 code : pieces) {
            const T* row = network->FeatureRow(F::Index(perspective, key, code));

            for (s00 i = 0; i < H; i++) values[i] = WrapAdd(values[i], row[i]);
        }

        return values;
    };

    alignas(Alignment) std::array<T, H> destination;

    s00 refreshes = 0;

    const auto refresh = [&](const u08 perspective, const u08 key, const std::vector<u16>& pieces) {
        const std::array<u64, 12> boards   = PieceBoards(pieces);
        const std::array<T  , H > expected = fresh(perspective, key, pieces);

        const auto view = [&](const u08 piece, const u08 color) { return boards[color * 6 + piece]; };

        for (s00 pass = 0; pass < 2; pass++) {
            destination.fill(T { 0x5A5A });

            table->Refresh(*network, perspective, key, view, destination.data());

            refreshes++;

            if (destination != expected)
                return testing::AssertionFailure() << "perspective " << s00 { perspective } << ", key " << s00 { key }
                                                   << ", " << pieces.size() << " pieces, pass " << pass;
        }

        return testing::AssertionSuccess();
    };

    std::vector<u16> shifted = StartPosition();

    for (u16& code : shifted) code ^= 16;

    for (u08 perspective = 0; perspective < 2         ; perspective++)
    for (u08 key         = 0; key         < F::KeyCount; key        ++) {
        ASSERT_TRUE(refresh(perspective, key, StartPosition()));
        ASSERT_TRUE(refresh(perspective, key, shifted        ));
        ASSERT_TRUE(refresh(perspective, key, {}             ));
    }

    Walk walk(random, true);

    walk.Reset(StartPosition());

    for (s00 trial = 0; trial < 2000; trial++) {
        if (trial % 64 == 0) walk.Reset(walk.RandomPosition());
        else                 walk.Next();

        const auto perspective = static_cast<u08>(random() % 2          );
        const auto key         = static_cast<u08>(random() % F::KeyCount);

        ASSERT_TRUE(refresh(perspective, key, walk.Pieces())) << "trial " << trial;
    }

    table->Clear(*network);

    for (u08 perspective = 0; perspective < 2         ; perspective++)
    for (u08 key         = 0; key         < F::KeyCount; key        ++) {
        ASSERT_TRUE(refresh(perspective, key, {}             ));
        ASSERT_TRUE(refresh(perspective, key, StartPosition()));
    }

    EXPECT_GT(refreshes, s00 { 4000 });
}

TEST(FinnyTable, ReloadRefreshesCache)
{
    FinnyTableTest::CheckReload<Examples::KingBucketed  , UpdatePolicy::Lazy >();
    FinnyTableTest::CheckReload<Examples::KingBucketed  , UpdatePolicy::Eager>();
    FinnyTableTest::CheckReload<Examples::OutputBucketed, UpdatePolicy::Lazy >();
}

TEST(FinnyTable, AssignedNetworkRefreshesCache)
{
    FinnyTableTest::CheckAssignment<UpdatePolicy::Lazy >();
    FinnyTableTest::CheckAssignment<UpdatePolicy::Eager>();
}

TEST(FinnyTable, StackBeforeFirstLoad)
{
    FinnyTableTest::CheckStackBeforeFirstLoad<UpdatePolicy::Lazy >();
    FinnyTableTest::CheckStackBeforeFirstLoad<UpdatePolicy::Eager>();
}

TEST(FinnyTable, LoadGeneration)
{
    using A = Examples::KingBucketed;

    static_assert(!FinnyTableTest::HasGeneration<Network<Examples::Flare>>);
    static_assert( FinnyTableTest::HasGeneration<Network<A              >>);

    std::mt19937_64 random(0x47454E4552415445);

    const std::vector<std::byte> bytes = RandomBytes<A>(random);

    const auto network = std::make_unique<Network<A>>();
    const auto other   = std::make_unique<Network<A>>();

    EXPECT_EQ(network->Generation(), u64 { 0 });

    ASSERT_TRUE(network->Load(bytes));

    const u64 first = network->Generation();

    EXPECT_NE(first, u64 { 0 });

    EXPECT_FALSE(network->Load(std::span(bytes).first(bytes.size() - 1)));
    EXPECT_EQ(network->Generation(), first);

    ASSERT_TRUE(other->Load(bytes));
    EXPECT_NE(other->Generation(), u64 { 0 });
    EXPECT_NE(other->Generation(), first);

    ASSERT_TRUE(network->Load(bytes));
    EXPECT_NE(network->Generation(), first);
    EXPECT_NE(network->Generation(), other->Generation());

    *other = *network;

    EXPECT_EQ(other->Generation(), network->Generation());
}

#endif //MANTARAY_RUNTIME_GTESTFINNYTABLE_H
