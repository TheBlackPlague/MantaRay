//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_EXAMPLES_EXAMPLE_H
#define MANTARAY_EXAMPLES_EXAMPLE_H

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <MantaRay/MantaRay.h>

#include "Position.h"

namespace Example
{

    using namespace MantaRay;

    class Random
    {

        u64 State;

        public:
        constexpr explicit Random(const u64 seed) : State(seed) {}

        constexpr u64 Next()
        {
            u64 value = State += 0x9E3779B97F4A7C15;

            value = (value ^ value >> 30) * 0xBF58476D1CE4E5B9;
            value = (value ^ value >> 27) * 0x94D049BB133111EB;

            return value ^ value >> 31;
        }

        constexpr f64 Unit() { return static_cast<f64>(Next() >> 11) / 9007199254740992.0; }

    };

    template<typename M, u16 I>
    void Fill(IO::Tensors<M>& tensors, Random& random)
    {
        using T = IO::Tensors<M>::template Element<I>;

        constexpr IR::Tensor Spec   = M::Graph.Tensors[I];
        constexpr Interval   Bounds = IR::BoundsOf(Spec);
        constexpr bool       Real   = IR::IsFloat(Spec.Type);
        constexpr bool       Clip   = Defined(Spec.ClipLow);

        constexpr f64 Low  = Clip ? ToDouble(Spec.ClipLow ) : -1.0;
        constexpr f64 High = Clip ? ToDouble(Spec.ClipHigh) :  1.0;

        for (T& value : tensors.template View<I>()) {
            if constexpr (Real) value = static_cast<T>(Low + (High - Low) * random.Unit());
            else {
                const u64 span = static_cast<u64>(Bounds.Max - Bounds.Min) + 1;

                value = static_cast<T>(Bounds.Min + static_cast<i64>(random.Next() % span));
            }
        }
    }

    template<auto Net>
    IO::Tensors<Model<Net>> SyntheticWeights(const u64 seed)
    {
        using M = Model<Net>;

        IO::Tensors<M> tensors {};
        Random         random (seed);

        [&]<u16... I>(const std::integer_sequence<u16, I...>) {
            (Fill<M, I>(tensors, random), ...);
        }(std::make_integer_sequence<u16, M::Graph.TensorCount> {});

        return tensors;
    }

    template<typename N, typename M>
    bool Load(N& network, const IO::Tensors<M>& tensors)
    {
        const std::vector<std::byte> file   = IO::Save(tensors);
        const LoadResult             result = network.Load(file);

        if (!result) {
            const std::string_view message = result.Message();

            std::printf("load failed: %.*s\n", static_cast<int>(message.size()), message.data());

            return false;
        }

        std::printf("weights: %zu bytes, synthetic\n\n", file.size());

        return true;
    }

    constexpr u08 SquareOf(const char (&name)[3]) { return static_cast<u08>(name[0] - 'a' + 8 * (name[1] - '1')); }

    struct Move
    {

        std::string_view Name;
        PieceDelta       Delta;

    };

    constexpr std::string_view Start = "r3k2r/1P1p2P1/8/4P3/8/8/8/R3K2R w KQkq - 0 1";

    constexpr std::array<Move, 12> Line = [] {
        std::array<Move, 12> line {};

        line[ 0].Name = "1. O-O-O";
        line[ 0].Delta.Move  (Piece::King , White, SquareOf("e1"), SquareOf("c1"));
        line[ 0].Delta.Move  (Piece::Rook , White, SquareOf("a1"), SquareOf("d1"));

        line[ 1].Name = "1... d5";
        line[ 1].Delta.Move  (Piece::Pawn , Black, SquareOf("d7"), SquareOf("d5"));

        line[ 2].Name = "2. exd6 e.p.";
        line[ 2].Delta.Remove(Piece::Pawn , Black, SquareOf("d5")                );
        line[ 2].Delta.Move  (Piece::Pawn , White, SquareOf("e5"), SquareOf("d6"));

        line[ 3].Name = "2... Rxh1";
        line[ 3].Delta.Remove(Piece::Rook , White, SquareOf("h1")                );
        line[ 3].Delta.Move  (Piece::Rook , Black, SquareOf("h8"), SquareOf("h1"));

        line[ 4].Name = "3. Rxh1";
        line[ 4].Delta.Remove(Piece::Rook , Black, SquareOf("h1")                );
        line[ 4].Delta.Move  (Piece::Rook , White, SquareOf("d1"), SquareOf("h1"));

        line[ 5].Name = "3... Kd7";
        line[ 5].Delta.Move  (Piece::King , Black, SquareOf("e8"), SquareOf("d7"));

        line[ 6].Name = "4. g8=Q";
        line[ 6].Delta.Remove(Piece::Pawn , White, SquareOf("g7")                );
        line[ 6].Delta.Insert(Piece::Queen, White, SquareOf("g8")                );

        line[ 7].Name = "4... Kc6";
        line[ 7].Delta.Move  (Piece::King , Black, SquareOf("d7"), SquareOf("c6"));

        line[ 8].Name = "5. bxa8=Q";
        line[ 8].Delta.Remove(Piece::Rook , Black, SquareOf("a8")                );
        line[ 8].Delta.Remove(Piece::Pawn , White, SquareOf("b7")                );
        line[ 8].Delta.Insert(Piece::Queen, White, SquareOf("a8")                );

        line[ 9].Name = "5... Kc5";
        line[ 9].Delta.Move  (Piece::King , Black, SquareOf("c6"), SquareOf("c5"));

        line[10].Name = "6. d7";
        line[10].Delta.Move  (Piece::Pawn , White, SquareOf("d6"), SquareOf("d7"));

        line[11].Name = "6... Kb4";
        line[11].Delta.Move  (Piece::King , Black, SquareOf("c5"), SquareOf("b4"));

        return line;
    }();

    inline Position Play(Position board, const PieceDelta& delta)
    {
        for (s00 i = 0; i < delta.RemovedCount(); i++) {
            const PieceChange change = delta.Removed(i);

            board.Remove(change.Side, change.Type, change.Square);
        }

        for (s00 i = 0; i < delta.InsertedCount(); i++) {
            const PieceChange change = delta.Inserted(i);

            board.Place(change.Side, change.Type, change.Square);
        }

        board.Pass();

        return board;
    }

    inline std::vector<Position> Boards()
    {
        std::vector<Position> boards { Position { Start } };

        for (const Move& move : Line) boards.push_back(Play(boards.back(), move.Delta));

        return boards;
    }

    inline void Print(const std::string_view what, const std::span<const i32> lanes, const Position& board)
    {
        std::printf("%-18.*s %c", static_cast<int>(what.size()), what.data(), board.SideToMove() == White ? 'w' : 'b');

        for (const i32 lane : lanes) std::printf(" %7d", lane);

        std::printf("\n");
    }

    inline void Print(const std::string_view what, const i32 value, const Position& board)
    {
        Print(what, std::span<const i32>(&value, 1), board);
    }

}

#endif //MANTARAY_EXAMPLES_EXAMPLE_H
