//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H
#define MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H

#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/Delta.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

template<typename S>
constexpr bool EvaluatesWithoutBoards = requires (S& stack) { stack.Evaluate(u08 {}); };

template<typename A>
constexpr bool FusedWhenLazy = Stack<A, UpdatePolicy::Lazy>::Fused && !Stack<A, UpdatePolicy::Eager>::Fused;

template<UpdatePolicy Policy>
void CheckResetAndDuplicates()
{
    using A = Examples::Starshard;

    std::mt19937_64 random(0x5245534554445550);

    const auto network = RandomNetwork<A>(random);
    const auto stack   = std::make_unique<Stack<A, Policy, 16>>(*network);

    Oracle<A> oracle(*network);

    std::vector<std::vector<u16>> positions { StartPosition() };

    const auto push = [&](const Delta& delta) {
        positions.push_back(Apply(positions.back(), delta));

        stack->Push(delta);
    };

    const auto pop = [&] {
        positions.pop_back();

        stack->Pop();
    };

    const auto check = [&](const u08 stm) {
        ASSERT_EQ(stack->Depth(), positions.size() - 1);

        EXPECT_TRUE(oracle.Check(*stack, positions.back(), stm)) << "depth " << stack->Depth();
    };

    const auto move = [](const u08 piece, const u08 color, const u08 from, const u08 to) {
        Delta delta;

        delta.Move(piece, color, from, to);

        return delta;
    };

    const auto capture = [](const u08 piece, const u08 color, const u08 from, const u08 to, const u08 victim) {
        Delta delta;

        delta.Remove(victim, color ^ 1, to  );
        delta.Move  (piece , color    , from, to);

        return delta;
    };

    ResetFromPieces(*stack, positions.back());

    check(0); check(0); check(1); check(0);

    push(move(0, 0, 12, 28));

    check(1); check(1); check(0);

    push(move   (0, 1, 51, 35));
    push(capture(0, 0, 28, 35, 0));

    check(1); check(1); check(0);

    pop();

    check(0); check(1);

    push(Delta {});

    check(1); check(0);

    push(move(1, 1, 62, 45));
    push(move(5, 0,  4, 12));
    push(Delta {});
    push(move(0, 1, 52, 36));

    check(0); check(1);

    pop(); pop();

    check(1); check(1);

    while (stack->Depth() > 0) pop();

    check(0); check(1);

    push(move(1, 0,  6, 21));
    push(move(0, 1, 52, 36));

    positions = { { PieceCode(5, 0, 4), PieceCode(3, 0, 7), PieceCode(5, 1, 60), PieceCode(0, 0, 49),
                    PieceCode(0, 1, 51), PieceCode(0, 0, 36), PieceCode(4, 1, 58) } };

    ResetFromPieces(*stack, positions.back());

    check(1); check(0);

    Delta castle;

    castle.Move(5, 0, 4, 6);
    castle.Move(3, 0, 7, 5);

    push(castle);
    push(move(0, 1, 51, 35));

    Delta passant;

    passant.Remove(0, 1, 35);
    passant.Move  (0, 0, 36, 43);

    push(passant);

    Delta promotion;

    promotion.Remove(0, 0, 49);
    promotion.Insert   (4, 0, 57);

    push(promotion);

    check(1); check(0);

    pop();

    Delta capturePromotion;

    capturePromotion.Remove(4, 1, 58);
    capturePromotion.Remove(0, 0, 49);
    capturePromotion.Insert   (3, 0, 58);

    push(capturePromotion);

    check(1); check(1); check(0);
}

template<UpdatePolicy Policy>
void CheckVectorAlignment()
{
    using A = Examples::Flare;
    using S = Stack<A, Policy, 247>;

    static_assert(alignof(S) == Alignment);

    std::mt19937_64 random(0x414C49474E4D454E);

    const auto network = RandomNetwork<A>(random);

    Oracle<A> oracle(*network);

    std::vector<S> stacks;

    for (s00 i = 0; i < 3; i++) {
        stacks.emplace_back(*network);

        for (const S& stack : stacks) {
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&stack                ) % Alignment, std::uintptr_t { 0 });
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(stack.Values(0).data()) % Alignment, std::uintptr_t { 0 });
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(stack.Values(1).data()) % Alignment, std::uintptr_t { 0 });
        }
    }

    const std::vector<u16> pieces = StartPosition();

    Delta delta;

    delta.Move(1, 0, 6, 21);

    const std::vector<u16> child = Apply(pieces, delta);

    for (S& stack : stacks) {
        ResetFromPieces(stack, pieces);

        stack.Push(delta);

        EXPECT_TRUE(oracle.Check(stack, child, 1));
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(stack.Values(0).data()) % Alignment, std::uintptr_t { 0 });
    }
}

TEST(AccumulatorStack, Shape)
{
    using Lazy  = AccumulatorStack<Examples::Flare, 247>;
    using Eager = AccumulatorStack<Examples::Flare, 247, UpdatePolicy::Eager>;

    static_assert(alignof(Lazy) == Alignment && alignof(Eager) == Alignment);
    static_assert(std::is_same_v<Lazy::Accumulator, i16>);
    static_assert(sizeof(Lazy) == 761024 && sizeof(Eager) == 761024);

    static_assert(std::is_same_v<Stack<Examples::WideAccumulator, UpdatePolicy::Lazy>::Accumulator, i32>);

    using N = Network<Examples::Flare>;

    static_assert( std::is_constructible_v<Lazy , const N& >);
    static_assert(!std::is_constructible_v<Lazy , const N&&>);
    static_assert(!std::is_constructible_v<Eager,       N  >);

    static_assert( FusedWhenLazy<Examples::Starshard        >);
    static_assert( FusedWhenLazy<Examples::Aurora           >);
    static_assert( FusedWhenLazy<Examples::Flare            >);
    static_assert( FusedWhenLazy<Examples::SimpleSCReLU     >);
    static_assert( FusedWhenLazy<Examples::WideSCReLU       >);
    static_assert( FusedWhenLazy<Examples::SquareFirstSCReLU>);
    static_assert( FusedWhenLazy<Examples::WidenedSCReLU    >);
    static_assert( FusedWhenLazy<Examples::LinearReLU       >);
    static_assert( FusedWhenLazy<Examples::WidenedSqrReLU   >);
    static_assert( FusedWhenLazy<Examples::OutputBucketed   >);
    static_assert( FusedWhenLazy<Examples::KingBucketed     >);
    static_assert(!Stack<Examples::MultiLayer     , UpdatePolicy::Lazy>::Fused);
    static_assert(!Stack<Examples::PackedPairwise , UpdatePolicy::Lazy>::Fused);
    static_assert(!Stack<Examples::ShiftedPairwise, UpdatePolicy::Lazy>::Fused);
    static_assert(!Stack<Examples::WideAccumulator, UpdatePolicy::Lazy>::Fused);

    static_assert( EvaluatesWithoutBoards<Stack<Examples::Flare         , UpdatePolicy::Lazy >>);
    static_assert( EvaluatesWithoutBoards<Stack<Examples::OutputBucketed, UpdatePolicy::Eager>>);
    static_assert(!EvaluatesWithoutBoards<Stack<Examples::KingBucketed  , UpdatePolicy::Lazy >>);
    static_assert(!EvaluatesWithoutBoards<Stack<Examples::KingBucketed  , UpdatePolicy::Eager>>);
    static_assert(!EvaluatesWithoutBoards<Stack<Examples::MultiLayer    , UpdatePolicy::Lazy >>);
}

TEST(AccumulatorStack, LazyEqualsEagerEqualsReset)
{
    LazyEqualsEagerEqualsReset<Examples::Starshard      >(0x4C415A5945414745, 3000);
    LazyEqualsEagerEqualsReset<Examples::Flare          >(0x464C415245574B31, 1000);
    LazyEqualsEagerEqualsReset<Examples::SimpleSCReLU   >(0x5343524555574B31, 3000);
    LazyEqualsEagerEqualsReset<Examples::WideSCReLU     >(0x574944454C5A5244, 1000);
    LazyEqualsEagerEqualsReset<Examples::WideAccumulator>(0x5749444541434331, 3000);
}

TEST(AccumulatorStack, EvaluateAfterResetAndDuplicates)
{
    CheckResetAndDuplicates<UpdatePolicy::Lazy >();
    CheckResetAndDuplicates<UpdatePolicy::Eager>();
}

TEST(AccumulatorStack, VectorAlignment)
{
    CheckVectorAlignment<UpdatePolicy::Lazy >();
    CheckVectorAlignment<UpdatePolicy::Eager>();
}

TEST(AccumulatorStackDeathTest, PopAtRoot)
{
#ifdef NDEBUG
    GTEST_SKIP() << "assertions are disabled";
#else
    using A = Examples::SimpleSCReLU;

    const auto network = std::make_unique<Network<A>>();
    const auto lazy    = std::make_unique<Stack<A, UpdatePolicy::Lazy , 4>>(*network);
    const auto eager   = std::make_unique<Stack<A, UpdatePolicy::Eager, 4>>(*network);

    EXPECT_DEATH(lazy ->Pop(), "Top > 0");
    EXPECT_DEATH(eager->Pop(), "Top > 0");

    lazy ->Push(Delta {});
    eager->Push(Delta {});
    lazy ->Pop();
    eager->Pop();

    EXPECT_DEATH(lazy ->Pop(), "Top > 0");
    EXPECT_DEATH(eager->Pop(), "Top > 0");
#endif
}

TEST(AccumulatorStackDeathTest, PushBeyondCapacity)
{
#ifdef NDEBUG
    GTEST_SKIP() << "assertions are disabled";
#else
    using A = Examples::SimpleSCReLU;

    const auto network = std::make_unique<Network<A>>();
    const auto lazy    = std::make_unique<Stack<A, UpdatePolicy::Lazy , 4>>(*network);
    const auto eager   = std::make_unique<Stack<A, UpdatePolicy::Eager, 4>>(*network);

    for (s00 i = 0; i < 3; i++) {
        lazy ->Push(Delta {});
        eager->Push(Delta {});
    }

    EXPECT_EQ(lazy ->Depth(), s00 { 3 });
    EXPECT_EQ(eager->Depth(), s00 { 3 });

    EXPECT_DEATH(lazy ->Push(Delta {}), "Top < Capacity - 1");
    EXPECT_DEATH(eager->Push(Delta {}), "Top < Capacity - 1");
#endif
}

#endif //MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H
