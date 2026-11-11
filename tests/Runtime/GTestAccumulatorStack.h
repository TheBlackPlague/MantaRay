//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H
#define MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/MantaRay.h>
#include <MantaRay/Reference/Position.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

namespace AccumulatorStackTest
{

    constexpr std::string_view Start = "r3k2r/1P1p2P1/8/4P3/8/8/8/R3K2R w KQkq - 0 1";

    constexpr std::array<s00, 7> Checked { 1, 2, 3, 5, 6, 9, 12 };

    constexpr u08 Square(const char (&name)[3]) { return static_cast<u08>(name[0] - 'a' + 8 * (name[1] - '1')); }

    constexpr std::array<PieceDelta, 12> Line = [] {
        std::array<PieceDelta, 12> line;

        line[ 0].Move  (Piece::King , White, Square("e1"), Square("c1"));
        line[ 0].Move  (Piece::Rook , White, Square("a1"), Square("d1"));

        line[ 1].Move  (Piece::Pawn , Black, Square("d7"), Square("d5"));

        line[ 2].Move  (Piece::Pawn , White, Square("e5"), Square("d6"));
        line[ 2].Remove(Piece::Pawn , Black, Square("d5")              );

        line[ 3].Remove(Piece::Rook , White, Square("h1")              );
        line[ 3].Move  (Piece::Rook , Black, Square("h8"), Square("h1"));

        line[ 4].Remove(Piece::Rook , Black, Square("h1")              );
        line[ 4].Move  (Piece::Rook , White, Square("d1"), Square("h1"));

        line[ 5].Move  (Piece::King , Black, Square("e8"), Square("d7"));

        line[ 6].Remove(Piece::Pawn , White, Square("g7")              );
        line[ 6].Insert(Piece::Queen, White, Square("g8")              );

        line[ 7].Move  (Piece::King , Black, Square("d7"), Square("c6"));

        line[ 8].Remove(Piece::Pawn , White, Square("b7")              );
        line[ 8].Remove(Piece::Rook , Black, Square("a8")              );
        line[ 8].Insert(Piece::Queen, White, Square("a8")              );

        line[ 9].Move  (Piece::King , Black, Square("c6"), Square("c5"));

        line[10].Move  (Piece::Pawn , White, Square("d6"), Square("d7"));

        line[11].Move  (Piece::King , Black, Square("c5"), Square("b4"));

        return line;
    }();

    constexpr std::array<PieceDelta, 3> Branch = [] {
        std::array<PieceDelta, 3> branch;

        branch[0].Remove(Piece::Pawn , White, Square("g7")              );
        branch[0].Insert(Piece::Queen, White, Square("g8")              );

        branch[1].Move  (Piece::King , Black, Square("e8"), Square("d7"));

        branch[2].Remove(Piece::Rook , Black, Square("a8")              );
        branch[2].Move  (Piece::Queen, White, Square("g8"), Square("a8"));

        return branch;
    }();

    inline void Apply(Reference::Position& position, const PieceDelta& delta)
    {
        for (s00 i = 0; i < delta.RemovedCount(); i++) {
            const PieceChange change = delta.Removed(i);

            position.Remove(change.Side, change.Type, change.Square);
        }

        for (s00 i = 0; i < delta.InsertedCount(); i++) {
            const PieceChange change = delta.Inserted(i);

            position.Place(change.Side, change.Type, change.Square);
        }

        position.Pass();
    }

    inline std::vector<Reference::Position> Played(
        const Reference::Position&        root  ,
        const std::span<const PieceDelta> deltas
    )
    {
        std::vector<Reference::Position> positions { root };

        for (const PieceDelta& delta : deltas) {
            Reference::Position next = positions.back();

            Apply(next, delta);

            positions.push_back(next);
        }

        return positions;
    }

    template<auto D, UpdatePolicy Policy>
    void CheckPolicy(const Network<D>& network)
    {
        const auto stack = std::make_unique<AccumulatorStack<D, 16, Policy>>(network);

        const std::vector<Reference::Position> trunk  = Played(Reference::Position { Start }, Line);
        const std::vector<Reference::Position> branch = Played(trunk[4], Branch);

        const auto expect = [&](const Reference::Position& position) {
            EXPECT_EQ(stack->Evaluate(position.SideToMove()), network.Evaluate(position))
                << "depth " << stack->Depth();
        };

        const auto pop = [&](const s00 depth) {
            while (stack->Depth() > depth) stack->Pop();

            expect(trunk[depth]);
        };

        stack->Reset(trunk[0]);

        for (s00 ply = 1; ply <= 12; ply++) {
            stack->Push(Line[ply - 1]);

            if (std::ranges::contains(Checked, ply)) expect(trunk[ply]);
        }

        pop(9);
        pop(4);

        for (s00 ply = 1; ply <= 3; ply++) {
            stack->Push(Branch[ply - 1]);

            if (ply != 2) expect(branch[ply]);
        }

        pop(0);

        for (s00 ply = 1; ply <= 12; ply++) stack->Push(Line[ply - 1]);

        expect(trunk[12]);
    }

    template<auto D>
    void Check(Random& random)
    {
        const auto network = std::make_unique<Network<D>>();

        const LoadResult result = network->Load(RandomBytes<D>(random));

        ASSERT_TRUE(result) << result.Message();

        CheckPolicy<D, UpdatePolicy::Eager>(*network);
        CheckPolicy<D, UpdatePolicy::Lazy >(*network);
    }

}

TEST(AccumulatorStack, Perspective)
{
    Random random(0x535441434B504552);

    AccumulatorStackTest::Check<Examples::Starshard>(random);
    AccumulatorStackTest::Check<Examples::Aurora   >(random);
    AccumulatorStackTest::Check<Examples::Flare    >(random);
}

#endif //MANTARAY_RUNTIME_GTESTACCUMULATORSTACK_H
