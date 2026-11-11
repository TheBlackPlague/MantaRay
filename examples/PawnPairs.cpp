//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include <memory>
#include <vector>

#include <MantaRay/MantaRay.h>

#include "Example.h"

namespace Networks
{

    constexpr static auto WithPawnPairs()
    {
        using namespace MantaRay;

        auto [psqStm, psqNtm] = Input<Chess768>();
        auto [ppStm , ppNtm ] = Input<PawnPairs<>>();

        auto psq  = Linear   (768 , 256, Weights<i16>(255, Clip(1.98)        ));
        auto pp   = Linear   (4560, 256, Weights<i08>(255, Clip(127.0 / 255)));
        auto bias = Parameter(      256, Biases <i16>(255, Clip(1.98)        ));

        const auto l1 = Affine(512, 1, Weights<i16>(64, Clip(1.98)), Biases<i16>(255 * 64));

        auto ft = [&](const auto& psqIn, const auto& ppIn) {
            return ClippedReLU(Wrap<i16>(psq(psqIn) + pp(ppIn) + bias));
        };

        return Output(l1(Concat(ft(psqStm, ppStm), ft(psqNtm, ppNtm))), 400);
    }

}

using PawnPairNetwork = MantaRay::Network<Networks::WithPawnPairs>;

static_assert(PawnPairNetwork::NeedsBoard && PawnPairNetwork::NeedsBoardAtEval);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::WithPawnPairs>(0x5041574E50414952);
    const auto network = std::make_unique<PawnPairNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::WithPawnPairs, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    const Example::Position root { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" };

    PieceDelta e4;
    PieceDelta d5;
    PieceDelta exd5;

    e4  .Move  (Piece::Pawn, White, Example::SquareOf("e2"), Example::SquareOf("e4"));
    d5  .Move  (Piece::Pawn, Black, Example::SquareOf("d7"), Example::SquareOf("d5"));
    exd5.Remove(Piece::Pawn, Black, Example::SquareOf("d5")                         );
    exd5.Move  (Piece::Pawn, White, Example::SquareOf("e4"), Example::SquareOf("d5"));

    const Example::Position afterE4   = Example::Play(root   , e4  );
    const Example::Position afterD5   = Example::Play(afterE4, d5  );
    const Example::Position afterExd5 = Example::Play(afterD5, exd5);

    stack->Reset(root);

    Example::Print("startpos", stack->Evaluate(root), root);

    stack->Push(e4  , afterE4  );
    stack->Push(d5  , afterD5  );
    stack->Push(exd5, afterExd5);

    Example::Print("2. exd5", stack->Evaluate(afterExd5), afterExd5);

    stack->Pop();

    Example::Print("back to 1... d5", stack->Evaluate(afterD5), afterD5);

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(boards[0]), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta, boards[ply]);

        if (ply % 3 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply]), boards[ply]);
    }

    return 0;
}
