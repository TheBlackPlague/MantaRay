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

    constexpr static auto OutputBucketed()
    {
        using namespace MantaRay;

        auto [stm, ntm] = Input<Chess768>();

        const auto l0 = Affine(768, 256, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
        const auto l1 =
            Affine(512, 8, Weights<i16>(64, Clip(1.98)).RowMajor(), Biases<i16>(255 * 64), DivideBy(255));

        const auto hidden = Concat(SquaredClippedReLU(l0(stm)), SquaredClippedReLU(l0(ntm)));

        return Output(Select<MaterialCount<8>>(l1(hidden)), 400);
    }

}

using OutputBucketNetwork = MantaRay::Network<Networks::OutputBucketed>;

static_assert(!OutputBucketNetwork::NeedsBoard && !OutputBucketNetwork::NeedsBoardAtEval);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::OutputBucketed>(0x4F55545055544255);
    const auto network = std::make_unique<OutputBucketNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::OutputBucketed, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(White), boards[0]);

    // Ten pieces is bucket 2. The en passant capture on ply 3 leaves nine, which is bucket 1.
    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply].SideToMove()), boards[ply]);
    }

    while (stack->Depth() > 2) stack->Pop();

    Example::Print("back to ply 2", stack->Evaluate(boards[2].SideToMove()), boards[2]);

    const Example::Position root { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" };

    stack->Reset(root);

    Example::Print("startpos", stack->Evaluate(White), root);

    return 0;
}
