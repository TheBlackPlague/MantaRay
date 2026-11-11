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

    constexpr static auto KingBucketed()
    {
        using namespace MantaRay;

        constexpr BucketLayout Buckets = Mirrored({
            0, 1, 2, 3,
            4, 4, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            8, 8, 8, 8,
            9, 9, 9, 9,
            9, 9, 9, 9
        });

        auto [stm, ntm] = Input<KingBuckets<Buckets>>();

        const auto l0 = Affine(7680, 256, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(0.99)));
        const auto l1 = Affine(512 , 1  , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64), DivideBy(255));

        return Output(l1(Concat(SquaredClippedReLU(l0(stm)), SquaredClippedReLU(l0(ntm)))), 400);
    }

}

using KingBucketNetwork = MantaRay::Network<Networks::KingBucketed>;

static_assert(!KingBucketNetwork::NeedsBoard && KingBucketNetwork::NeedsBoardAtEval);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::KingBucketed>(0x4B494E474255434B);
    const auto network = std::make_unique<KingBucketNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::KingBucketed, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(boards[0]), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply]), boards[ply]);
    }

    while (stack->Depth() > 1) stack->Pop();

    Example::Print("back to ply 1", stack->Evaluate(boards[1]), boards[1]);

    stack->ClearCache();

    for (s00 ply = 2; ply <= 8; ply++) stack->Push(Example::Line[ply - 1].Delta);

    Example::Print("ply 8, cold cache", stack->Evaluate(boards[8]), boards[8]);

    return 0;
}
