//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#include <cstdio>
#include <memory>
#include <vector>

#include <MantaRay/MantaRay.h>

#include "Example.h"

namespace Networks
{

    constexpr static auto BucketsPairsThreats()
    {
        using namespace MantaRay;

        constexpr BucketLayout Buckets = Mirrored({
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 8, 8, 8,
            9, 9, 9, 9,
            9, 9, 9, 9,
            9, 9, 9, 9,
            9, 9, 9, 9,
            9, 9, 9, 9
        });

        auto [psqStm, psqNtm] = Input<KingBuckets<Buckets>>();
        auto [ ppStm,  ppNtm] = Input<PawnPairs<>>();
        auto [thrStm, thrNtm] = Input<Threats<TI59808>>();

        auto psq  = Linear   (7680 , 256, Weights<i16>(255, Clip(1.98)       ));
        auto tipp = Linear   (64368, 256, Weights<i08>(255, Clip(127.0 / 255)));
        auto bias = Parameter(       256, Biases <i16>(255, Clip(0.99)       ));

        auto l1 = Affine(512, 1, Weights<i16>(64, Clip(1.98)), Biases<i16>(255 * 64));

        auto ft = [&](const auto& psqIn, const auto& ppIn, const auto& thrIn) {
            return ClippedReLU(
                Wrap<i16>(psq(psqIn) + tipp.Inputs(0, 4560)(ppIn) + tipp.Inputs(4560, 64368)(thrIn) + bias)
            );
        };

        return Output(l1(Concat(ft(psqStm, ppStm, thrStm), ft(psqNtm, ppNtm, thrNtm))), 400);
    }

}

using SplitStack    = MantaRay::AccumulatorStack<Networks::BucketsPairsThreats, 32>;
using CombinedStack = MantaRay::AccumulatorStack<
    Networks::BucketsPairsThreats, 32, MantaRay::UpdatePolicy::Lazy, MantaRay::Accumulators::Combined
>;

int main()
{
    using namespace MantaRay;

    const auto tensors = Example::SyntheticWeights<Networks::BucketsPairsThreats>(0x434F4D42494E4544);
    const auto network = std::make_unique<Network<Networks::BucketsPairsThreats>>();

    if (!Example::Load(*network, tensors)) return 1;

    // Both stacks run on the same network.
    const auto split    = std::make_unique<SplitStack   >(*network);
    const auto combined = std::make_unique<CombinedStack>(*network);

    std::printf("stack: %zu bytes split, %zu bytes combined\n\n", sizeof(SplitStack), sizeof(CombinedStack));

    const std::vector<Example::Position> boards = Example::Boards();

    split   ->Reset(boards[0]);
    combined->Reset(boards[0]);

    Example::Print("start, split"   , split   ->Evaluate(boards[0]), boards[0]);
    Example::Print("start, combined", combined->Evaluate(boards[0]), boards[0]);

    for (s00 ply = 1; ply <= Example::Line.size(); ply++) {
        split   ->Push(Example::Line[ply - 1].Delta, boards[ply]);
        combined->Push(Example::Line[ply - 1].Delta, boards[ply]);

        if (ply % 4 != 0) continue;

        Example::Print(Example::Line[ply - 1].Name, split   ->Evaluate(boards[ply]), boards[ply]);
        Example::Print("  combined"               , combined->Evaluate(boards[ply]), boards[ply]);
    }

    while (split   ->Depth() > 6) split   ->Pop();
    while (combined->Depth() > 6) combined->Pop();

    Example::Print("back to ply 6", split   ->Evaluate(boards[6]), boards[6]);
    Example::Print("  combined"   , combined->Evaluate(boards[6]), boards[6]);

    return 0;
}
