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

    constexpr static auto SquaredPerspective()
    {
        using namespace MantaRay;

        auto [stm, ntm] = Input<Chess768>();

        const auto l0 = Affine(768, 256, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));

        const auto l1 = Affine(512, 1  , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64), DivideBy(255));

        const auto y0 = SquaredClippedReLU(l0(stm));
        const auto y1 = SquaredClippedReLU(l0(ntm));

        return Output(l1(Concat(y0, y1)), 400);
    }

}

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::SquaredPerspective>(0x5351554152454421);
    const auto network = std::make_unique<Network<Networks::SquaredPerspective>>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::SquaredPerspective, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(White), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        if (ply % 2 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply].SideToMove()), boards[ply]);
    }

    while (stack->Depth() > 5) stack->Pop();

    Example::Print("back to ply 5", stack->Evaluate(boards[5].SideToMove()), boards[5]);

    Example::Print("from scratch", network->Evaluate(boards[5]), boards[5]);

    return 0;
}
