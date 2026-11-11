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

    constexpr static auto Perspective()
    {
        using namespace MantaRay;

        auto [stm, ntm] = Input<Chess768>();

        const auto l0 = Affine(768, 256, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
        const auto l1 = Affine(512, 1  , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64       ));

        const auto y0 = ClippedReLU(l0(stm));
        const auto y1 = ClippedReLU(l0(ntm));

        return Output(l1(Concat(y0, y1)), 400);
    }

}

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::Perspective>(0x5045525350);
    const auto network = std::make_unique<Network<Networks::Perspective>>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::Perspective, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(White), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        if (ply % 3 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply].SideToMove()), boards[ply]);
    }

    while (stack->Depth() > 4) stack->Pop();

    Example::Print("back to ply 4", stack->Evaluate(boards[4].SideToMove()), boards[4]);

    stack->Pop();
    stack->Pop();

    Example::Print("back to ply 2", stack->Evaluate(boards[2].SideToMove()), boards[2]);

    const Example::Position root { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" };

    PieceDelta e4;

    e4.Move(Piece::Pawn, White, Example::SquareOf("e2"), Example::SquareOf("e4"));

    stack->Reset(root);

    Example::Print("startpos", stack->Evaluate(White), root);

    stack->Push(e4);

    Example::Print("1. e4", stack->Evaluate(Black), Example::Play(root, e4));

    stack->Pop();

    Example::Print("startpos again", stack->Evaluate(White), root);

    return 0;
}
