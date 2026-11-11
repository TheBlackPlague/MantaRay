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

    constexpr static auto Heads()
    {
        using namespace MantaRay;

        using Stack = MaterialCount<8>;

        auto [stm, ntm] = Input<Chess768>();

        auto l0  = Affine(768, 256   , Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
        auto l1  = Affine(256, 8 * 16, Weights<i08>(64 , Clip(1.98)), Biases<f32>(), Dequantize());
        auto l2  = Affine(15 , 8 * 32, Weights<f32>(), Biases<f32>());
        auto l3  = Affine(32 , 8     , Weights<f32>(), Biases<f32>());
        auto wdl = Affine(32 , 3     , Weights<f32>(), Biases<f32>());

        auto ft = [&](const auto& x) { return Pairwise(ClippedReLU(l0(x))) >> 9; };

        // The first hidden layer has 16 outputs: 15 feed the next layer, the last one skips straight to the output.
        auto y    = Select<Stack>(l1(Concat(ft(stm), ft(ntm))));
        auto main = Slice(y, 0 , 15);
        auto skip = Slice(y, 15, 16);
        auto z    = SquaredClippedReLU(Select<Stack>(l2(ClippedReLU(main))));

        // The file spec lists the tensors in the order Bullet writes them.
        return Output(
            Bullet(l0.Weights(), l0.Biases(), l1.Weights(), l1.Biases(), l2.Weights(), l2.Biases(), l3.Weights(),
                   l3.Biases(), wdl.Weights(), wdl.Biases()),
            Head { Select<Stack>(l3(z)) + skip, 400 },
            Head { wdl(z)                     , 100 }
        );
    }

}

using HeadsNetwork = MantaRay::Network<Networks::Heads>;

static_assert(HeadsNetwork::Outputs == 4);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::Heads>(0x4845414453574443);
    const auto network = std::make_unique<HeadsNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::Heads, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", std::span<const i32>(stack->EvaluateAll(boards[0])), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        if (ply % 3 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, std::span<const i32>(stack->EvaluateAll(boards[ply])), boards[ply]);
    }

    while (stack->Depth() > 4) stack->Pop();

    Example::Print("back to ply 4", std::span<const i32>(stack->EvaluateAll(boards[4])), boards[4]);

    // Evaluate is the first lane, the evaluation head.
    Example::Print("evaluation only", stack->Evaluate(boards[4]), boards[4]);

    return 0;
}
