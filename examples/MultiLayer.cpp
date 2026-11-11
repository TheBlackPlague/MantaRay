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

    constexpr static auto MultiLayer()
    {
        using namespace MantaRay;

        using Stack = MaterialCount<8>;

        auto [stm, ntm] = Input<Chess768>();

        auto l0 = Affine(768, 512   , Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98))               );
        auto l1 = Affine(512, 8 * 16, Weights<i08>(64).RowMajor()  , Biases<f32>(               ),  Dequantize());
        auto l2 = Affine(16 , 8 * 32, Weights<f32>(  ).RowMajor()  , Biases<f32>()                              );
        auto l3 = Affine(32 , 8     , Weights<f32>(  ).RowMajor()  , Biases<f32>()                              );

        // Pairwise multiplication of two halves, scaled back to [0, 127] by the shift
        auto ft = [&](const auto& x) { return Pairwise(ClippedReLU(l0(x))) >> 9; };

        auto h1 = SquaredClippedReLU(Select<Stack>(l1(Concat(ft(stm), ft(ntm)))));
        auto h2 = SquaredClippedReLU(Select<Stack>(l2(h1)));

        return Output(Select<Stack>(l3(h2)), 400);
    }

}

using MultiLayerNetwork = MantaRay::Network<Networks::MultiLayer>;

static_assert(!MultiLayerNetwork::Fusable);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::MultiLayer>(0x4D554C54494C4159);
    const auto network = std::make_unique<MultiLayerNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::MultiLayer, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(White), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta);

        if (ply % 2 != 1) continue;

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply].SideToMove()), boards[ply]);
    }

    while (stack->Depth() > 6) stack->Pop();

    Example::Print("back to ply 6", stack->Evaluate(boards[6].SideToMove()), boards[6]);

    return 0;
}
