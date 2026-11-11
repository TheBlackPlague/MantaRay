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

    constexpr static auto WithThreats()
    {
        using namespace MantaRay;

        auto [psqStm, psqNtm] = Input<Chess768>();
        auto [thrStm, thrNtm] = Input<Threats<TI59808>>();

        auto psq  = Linear   (  768, 256, Weights<i16>(255, Clip(1.98)       ));
        auto thr  = Linear   (59808, 256, Weights<i08>(255, Clip(127.0 / 255)));
        auto bias = Parameter(       256, Biases <i16>(255, Clip(1.98)       ));

        const auto l1 = Affine(512, 1, Weights<i16>(64, Clip(1.98)), Biases<i16>(255 * 64));

        auto ft = [&](const auto& psqIn, const auto& thrIn) {
            return ClippedReLU(Wrap<i16>(psq(psqIn) + thr(thrIn) + bias));
        };

        return Output(l1(Concat(ft(psqStm, thrStm), ft(psqNtm, thrNtm))), 400);
    }

}

using ThreatNetwork = MantaRay::Network<Networks::WithThreats>;

static_assert(ThreatNetwork::NeedsBoard && ThreatNetwork::NeedsBoardAtEval);

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::WithThreats>(0x5448524541545321);
    const auto network = std::make_unique<ThreatNetwork>();

    if (!Example::Load(*network, tensors)) return 1;

    const auto stack = std::make_unique<AccumulatorStack<Networks::WithThreats, 32>>(*network);

    const std::vector<Example::Position> boards = Example::Boards();

    stack->Reset(boards[0]);

    Example::Print("start", stack->Evaluate(boards[0]), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        stack->Push(Example::Line[ply - 1].Delta, boards[ply]);

        if (ply % 2 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, stack->Evaluate(boards[ply]), boards[ply]);
    }

    while (stack->Depth() > 3) stack->Pop();

    Example::Print("back to ply 3", stack->Evaluate(boards[3]), boards[3]);

    return 0;
}
