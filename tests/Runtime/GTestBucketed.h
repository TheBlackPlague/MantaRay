//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTBUCKETED_H
#define MANTARAY_RUNTIME_GTESTBUCKETED_H

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Runtime/AccumulatorStack.h>
#include <MantaRay/Runtime/Delta.h>
#include <MantaRay/Runtime/Network.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

namespace BucketedTest
{

    using Keyed = Examples::KingBucketed::Transformer::Features;

    static_assert(Keyed::Key(0,  4) == 3 && Keyed::Key(1, 60) == 3);
    static_assert(Keyed::Key(0,  5) == 3 && Keyed::Key(1, 61) == 3);
    static_assert(Keyed::Key(0,  3) == 2 && Keyed::Key(1, 59) == 2);
    static_assert(Keyed::Key(0, 12) == 5 && Keyed::Key(1, 52) == 5);
    static_assert(Keyed::Key(0, 11) == 4 && Keyed::Key(1, 51) == 4);
    static_assert(Keyed::Key(0,  6) == 1 && Keyed::Key(1, 62) == 1);

    inline Delta Move(const u08 piece, const u08 color, const u08 from, const u08 to)
    {
        Delta delta;

        delta.Move(piece, color, from, to);

        return delta;
    }

    inline std::vector<u16> OpenPosition()
    {
        std::vector<u16> pieces;

        for (u08 color = 0; color < 2; color++) {
            const u08 flip = color * 56;

            pieces.push_back(PieceCode(KingPiece, color,  4 ^ flip));
            pieces.push_back(PieceCode(3        , color,  0 ^ flip));
            pieces.push_back(PieceCode(3        , color,  7 ^ flip));
            pieces.push_back(PieceCode(4        , color, 18 ^ flip));

            for (const u08 square : { 8, 9, 10, 14, 15 }) pieces.push_back(PieceCode(0, color, square ^ flip));
        }

        return pieces;
    }

    inline std::vector<Delta> Fillers()
    {
        constexpr std::array<std::array<u08, 3>, 5> Moves {{
            { 0, 8, 16 }, { 0, 9, 17 }, { 0, 14, 22 }, { 0, 15, 23 }, { 4, 18, 26 }
        }};

        std::vector<Delta> fillers;

        for (const auto& [piece, from, to] : Moves)
        for (u08 color = 0; color < 2; color++)
            fillers.push_back(Move(piece, color, from ^ color * 56, to ^ color * 56));

        return fillers;
    }

    inline Delta KingMove(const u08 color, const u08 from, const u08 to)
    {
        return Move(KingPiece, color, from ^ color * 56, to ^ color * 56);
    }

    inline Delta Castle(const u08 color, const bool rookFirst)
    {
        Delta delta;

        if (rookFirst) delta.Move(3, color, 7 ^ color * 56, 5 ^ color * 56);

        delta.Move(KingPiece, color, 4 ^ color * 56, 6 ^ color * 56);

        if (!rookFirst) delta.Move(3, color, 7 ^ color * 56, 5 ^ color * 56);

        return delta;
    }

    template<UpdatePolicy Policy>
    class Line
    {

        using A = Examples::KingBucketed;

        std::unique_ptr<Stack<A, Policy, 16>> Under;

        Oracle<A> Reference;

        std::vector<std::vector<u16>> Positions;

        public:
        s00 Checks = 0;

        explicit Line(const Network<A>& network) :
            Under(std::make_unique<Stack<A, Policy, 16>>(network)), Reference(network) {}

        void Reset()
        {
            Positions = { OpenPosition() };

            ResetFromPieces(*Under, Positions.back());
        }

        void Push(const Delta& delta)
        {
            Positions.push_back(Apply(Positions.back(), delta));

            Under->Push(delta);
        }

        void Pop()
        {
            Positions.pop_back();

            Under->Pop();
        }

        void ClearCache() { Under->ClearCache(); }

        s00 Depth() const { return Under->Depth(); }

        testing::AssertionResult Check()
        {
            Checks++;

            if (Under->Depth() != Positions.size() - 1) return testing::AssertionFailure() << "depth mismatch";

            for (u08 stm = 0; stm < 2; stm++)
                if (testing::AssertionResult result = Reference.Check(*Under, Positions.back(), stm); !result)
                    return result << " at depth " << Under->Depth();

            return testing::AssertionSuccess();
        }

        testing::AssertionResult Unwind()
        {
            while (Under->Depth() > 0) {
                Pop();

                if (testing::AssertionResult result = Check(); !result) return result << " while unwinding";
            }

            return testing::AssertionSuccess();
        }

    };

    template<UpdatePolicy Policy>
    void CheckKingCrossings()
    {
        using A = Examples::KingBucketed;

        constexpr std::array<u08, 4> Targets { 5, 3, 12, 11 };

        std::mt19937_64 random(0x4B494E4743524F53);

        const auto network = RandomNetwork<A>(random);

        Line<Policy> line(*network);

        const std::vector<Delta> fillers = Fillers();

        s00 scenario = 0;

        const auto begin = [&] {
            if (scenario++ % 7 == 3) line.ClearCache();

            line.Reset();
        };

        const auto play = [&](const s00 depth, const auto& special) {
            s00 filler = 0;

            for (s00 ply = 1; ply <= depth; ply++) {
                const Delta* delta = special(ply);

                line.Push(delta != nullptr ? *delta : fillers[filler++]);
            }
        };

        for (s00 depth = 1; depth <= 5; depth++)
        for (s00 at    = 1; at <= depth; at++)
        for (u08 color = 0; color < 2; color++)
        for (s00 kind  = 0; kind < Targets.size() + 2; kind++) {
            const Delta first   = kind < Targets.size() ? KingMove(color, 4, Targets[kind])
                                                        : Castle(color, kind > Targets.size());
            const Delta sibling = KingMove(color, 4, Targets[(kind + 1) % Targets.size()]);

            const std::string where = "depth " + std::to_string(depth) + ", king move at " + std::to_string(at) +
                                      ", color " + std::to_string(color) + ", kind " + std::to_string(kind);

            begin();
            play(depth, [&](const s00 ply) { return ply == at ? &first : nullptr; });

            ASSERT_TRUE(line.Check ()) << where;
            ASSERT_TRUE(line.Unwind()) << where;

            begin();
            play(depth, [&](const s00 ply) { return ply == at ? &first : nullptr; });

            ASSERT_TRUE(line.Check()) << where << ", before sibling";

            while (line.Depth() >= at) line.Pop();

            line.Push(sibling);

            for (s00 ply = at + 1; ply <= depth; ply++) line.Push(fillers[ply]);

            ASSERT_TRUE(line.Check ()) << where << ", sibling";
            ASSERT_TRUE(line.Unwind()) << where << ", sibling";
        }

        for (s00 depth = 2; depth <= 5; depth++)
        for (s00 white = 1; white <= depth; white++)
        for (s00 black = 1; black <= depth; black++) {
            if (white == black) continue;

            const Delta whiteMove = KingMove(0, 4, 11);
            const Delta blackMove = KingMove(1, 4, 12);

            begin();
            play(depth, [&](const s00 ply) {
                return ply == white ? &whiteMove : ply == black ? &blackMove : nullptr;
            });

            const std::string where = "both kings, depth " + std::to_string(depth) + ", white at " +
                                      std::to_string(white) + ", black at " + std::to_string(black);

            ASSERT_TRUE(line.Check ()) << where;
            ASSERT_TRUE(line.Unwind()) << where;
        }

        for (s00 depth = 2; depth <= 5; depth++)
        for (s00 at    = 1; at < depth; at++)
        for (u08 color = 0; color < 2; color++) {
            const Delta away = KingMove(color, 4, 3);
            const Delta back = KingMove(color, 3, 4);

            begin();
            play(depth, [&](const s00 ply) { return ply == at ? &away : ply == at + 1 ? &back : nullptr; });

            const std::string where = "away and back, depth " + std::to_string(depth) + ", at " +
                                      std::to_string(at) + ", color " + std::to_string(color);

            ASSERT_TRUE(line.Check ()) << where;
            ASSERT_TRUE(line.Unwind()) << where;
        }

        begin();

        line.Push(KingMove(0, 4, 3));

        ASSERT_TRUE(line.Check());

        line.Push(fillers[1]);
        line.Push(KingMove(0, 3, 11));
        line.Push(fillers[0]);

        ASSERT_TRUE(line.Check());

        line.Push(KingMove(1, 4, 12));
        line.Push(KingMove(0, 11, 12));

        ASSERT_TRUE(line.Check ()) << "chained crossings";
        ASSERT_TRUE(line.Unwind()) << "chained crossings";

        EXPECT_GT(line.Checks, s00 { 1000 });
    }

    inline void CheckEagerUnaffectedPerspective()
    {
        using A = Examples::KingBucketed;

        std::mt19937_64 random(0x554E414646454354);

        const auto network = RandomNetwork<A>(random);
        const auto stack   = std::make_unique<Stack<A, UpdatePolicy::Eager, 8>>(*network);
        const auto fresh   = std::make_unique<Stack<A, UpdatePolicy::Lazy , 1>>(*network);

        const std::vector<Delta> fillers = Fillers();

        std::vector<u16> pieces = OpenPosition();

        ResetFromPieces(*stack, pieces);

        const auto push = [&](const Delta& delta) {
            pieces = Apply(pieces, delta);

            stack->Push(delta);
        };

        const auto same = [&](const u08 perspective) {
            fresh->ClearCache();

            ResetFromPieces(*fresh, pieces);

            return SameValues(*stack, std::span(fresh->Values(perspective)), perspective);
        };

        push(KingMove(0, 4, 11));

        EXPECT_TRUE(same(1)) << "black advanced while the white refresh is deferred";

        push(fillers[1]);

        EXPECT_TRUE(same(1)) << "black keeps advancing";

        push(KingMove(1, 4, 12));

        EXPECT_EQ(EvaluateWithBoards(*stack, pieces, 0), network->Evaluate(pieces, 0));

        push(fillers[0]);

        EXPECT_TRUE(same(0)) << "white advanced after the deferred refresh";
        EXPECT_TRUE(same(1)) << "black advanced after the deferred refresh";
    }

    template<typename A, UpdatePolicy Policy>
    void CheckPieceCounts(const u64 seed)
    {
        using B = Network<A>::Buckets;

        static_assert(B::Count == 8);

        std::mt19937_64 random(seed);

        const auto network = RandomNetwork<A>(random);
        const auto stack   = std::make_unique<Stack<A, Policy, 64>>(*network);

        std::vector<std::vector<u16>> positions { StartPosition() };

        ResetFromPieces(*stack, positions.back());

        const auto push = [&](const Delta& delta) {
            positions.push_back(Apply(positions.back(), delta));

            stack->Push(delta);
        };

        const auto pick = [&](const u08 color, const bool kings) {
            std::vector<u16> found;

            for (const u16 code : positions.back())
                if (ColorOf(code) == color && (kings || PieceOf(code) != KingPiece)) found.push_back(code);

            return found.empty() ? u16 { 0xFFFF } : found[random() % found.size()];
        };

        std::vector<s00> buckets;

        const auto check = [&](const u08 stm) -> testing::AssertionResult {
            const std::vector<u16>& pieces = positions.back();

            const s00 bucket = B::Of(static_cast<u08>(pieces.size()));

            i32 actual = 0;

            if constexpr (Bucketed<A>) actual = EvaluateWithBoards(*stack, pieces, stm);
            else                       actual = stack->Evaluate(stm);

            std::array<i32, B::Count> outputs {};

            for (s00 b = 0; b < B::Count; b++)
                outputs[b] = network->Forward(stack->Values(stm), stack->Values(stm ^ 1), b);

            buckets.push_back(bucket);

            if (actual != outputs[bucket])
                return testing::AssertionFailure() << pieces.size() << " pieces: " << actual << " is not bucket "
                                                   << bucket << " (" << outputs[bucket] << ")";

            if (std::ranges::count(outputs, actual) != 1)
                return testing::AssertionFailure() << pieces.size() << " pieces: buckets are not distinguishable";

            if (actual != network->Evaluate(pieces, stm))
                return testing::AssertionFailure() << pieces.size() << " pieces: differs from the network";

            return testing::AssertionSuccess();
        };

        ASSERT_TRUE(check(0));

        for (s00 step = 0; positions.back().size() > 2; step++) {
            const auto side = static_cast<u08>(step % 2);

            u16 victim = pick(side ^ 1, false);

            if (victim == 0xFFFF) victim = pick(side, false);

            const u16 mover = pick(ColorOf(victim) ^ 1, true);

            Delta capture;

            capture.Remove(PieceOf(victim), ColorOf(victim), SquareOf(victim));
            capture.Move  (PieceOf(mover ), ColorOf(mover ), SquareOf(mover ), SquareOf(victim));

            push(capture);

            if (step % 5 == 2) {
                const u16 pawn = pick(side, false);

                if (pawn != 0xFFFF) {
                    Delta promotion;

                    promotion.Remove(PieceOf(pawn), side, SquareOf(pawn));
                    promotion.Insert   (4            , side, SquareOf(pawn));

                    push(promotion);
                    push(Delta {});
                }
            }

            if (Policy == UpdatePolicy::Eager || step % 3 != 1) ASSERT_TRUE(check(step % 2)) << "step " << step;
        }

        ASSERT_TRUE(check(1));

        while (stack->Depth() > 0) {
            positions.pop_back();

            stack->Pop();

            ASSERT_TRUE(check(stack->Depth() % 2)) << "unwinding at depth " << stack->Depth();
        }

        s00 changes = 0;

        for (s00 i = 1; i < buckets.size(); i++) changes += buckets[i] != buckets[i - 1];

        EXPECT_EQ(buckets.front(), s00 { 7 });
        EXPECT_GE(changes, s00 { 14 });
    }

    template<typename A, UpdatePolicy Policy>
    void CheckClampedPieceCounts()
    {
        using B = Network<A>::Buckets;

        std::mt19937_64 random(0x434C414D50454430);

        const auto network = RandomNetwork<A>(random);
        const auto stack   = std::make_unique<Stack<A, Policy, 4>>(*network);

        const auto forward = [&](const s00 bucket) {
            return network->Forward(stack->Values(0), stack->Values(1), bucket);
        };

        const std::vector<u16> lone { PieceCode(0, 0, 12) };

        ResetFromPieces(*stack, lone);

        EXPECT_EQ(stack->Evaluate(0), forward(B::Of(2)));

        ResetFromPieces(*stack, std::vector<u16> {});

        EXPECT_EQ(stack->Evaluate(0), forward(B::Of(2)));

        ResetFromPieces(*stack, StartPosition());

        Delta extra;

        extra.Insert(4, 0, 30);
        extra.Insert(4, 0, 31);

        stack->Push(extra);

        EXPECT_EQ(stack->Evaluate(0), forward(B::Of(32)));
    }

}

TEST(Bucketed, LazyEqualsEagerEqualsReset)
{
    LazyEqualsEagerEqualsReset<Examples::KingBucketed  >(0x4B42344852414E44, 3000);
    LazyEqualsEagerEqualsReset<Examples::OutputBucketed>(0x4F42385241414E44, 3000);
    LazyEqualsEagerEqualsReset<Examples::MultiLayer    >(0x4D554C5449524E44, 2000);
}

TEST(Bucketed, KingCrossings)
{
    BucketedTest::CheckKingCrossings<UpdatePolicy::Lazy >();
    BucketedTest::CheckKingCrossings<UpdatePolicy::Eager>();
}

TEST(Bucketed, EagerAdvancesUnaffectedPerspective)
{
    BucketedTest::CheckEagerUnaffectedPerspective();
}

TEST(Bucketed, PieceCountTracking)
{
    BucketedTest::CheckPieceCounts<Examples::OutputBucketed, UpdatePolicy::Lazy >(0x504945434553304C);
    BucketedTest::CheckPieceCounts<Examples::OutputBucketed, UpdatePolicy::Eager>(0x5049454345533045);
    BucketedTest::CheckPieceCounts<Examples::KingBucketed  , UpdatePolicy::Lazy >(0x504945434553314C);
    BucketedTest::CheckPieceCounts<Examples::KingBucketed  , UpdatePolicy::Eager>(0x5049454345533145);
}

TEST(Bucketed, PieceCountClamped)
{
#ifndef NDEBUG
    GTEST_SKIP() << "out-of-range piece counts assert in debug builds";
#else
    BucketedTest::CheckClampedPieceCounts<Examples::OutputBucketed, UpdatePolicy::Lazy >();
    BucketedTest::CheckClampedPieceCounts<Examples::OutputBucketed, UpdatePolicy::Eager>();
#endif
}

TEST(BucketedDeathTest, PieceCountOutOfRange)
{
#ifdef NDEBUG
    GTEST_SKIP() << "assertions are disabled";
#else
    using A = Examples::OutputBucketed;

    std::mt19937_64 random(0x4445415448434E54);

    const auto network = RandomNetwork<A>(random);
    const auto lazy    = std::make_unique<Stack<A, UpdatePolicy::Lazy , 4>>(*network);
    const auto eager   = std::make_unique<Stack<A, UpdatePolicy::Eager, 4>>(*network);

    const std::vector<u16> lone { PieceCode(0, 0, 12) };

    ResetFromPieces(*lazy , lone);
    ResetFromPieces(*eager, lone);

    EXPECT_DEATH((void) lazy ->Evaluate(0), "pieces >= MinPieces");
    EXPECT_DEATH((void) eager->Evaluate(0), "pieces >= MinPieces");

    Delta extra;

    extra.Insert(4, 0, 30);
    extra.Insert(4, 0, 31);

    ResetFromPieces(*lazy , StartPosition());
    ResetFromPieces(*eager, StartPosition());

    lazy ->Push(extra);
    eager->Push(extra);

    EXPECT_DEATH((void) lazy ->Evaluate(1), "pieces <= MaxPieces");
    EXPECT_DEATH((void) eager->Evaluate(1), "pieces <= MaxPieces");
#endif
}

#endif //MANTARAY_RUNTIME_GTESTBUCKETED_H
