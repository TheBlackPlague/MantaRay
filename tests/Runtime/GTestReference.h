//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTREFERENCE_H
#define MANTARAY_RUNTIME_GTESTREFERENCE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/MantaRay.h>
#include <MantaRay/Reference/Position.h>

#include "../Architecture/Examples.h"
#include "NetworkUtil.h"

using namespace MantaRay;

namespace ReferenceTest
{

    constexpr std::array<std::string_view, 8> Fens {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 4",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    };

    template<auto D>
    i32 ReferenceEvaluate(const std::span<const std::byte> bytes, const Reference::Position& position)
    {
        constexpr s00 Width  = IR::ElementsOf(Model<D>::Graph.Tensors[1]);
        constexpr s00 Inputs = 768;
        constexpr s00 Empty  = 12;

        constexpr s00 BiasBegin   = Inputs * Width;
        constexpr s00 WeightBegin = BiasBegin + Width;
        constexpr s00 OffsetBegin = WeightBegin + 2 * Width;

        const auto read = [&](const s00 index) {
            const u16 low  = std::to_integer<u16>(bytes[2 * index    ]);
            const u16 high = std::to_integer<u16>(bytes[2 * index + 1]);

            return i64 { static_cast<i16>(low | high << 8) };
        };

        const u08 stm = position.SideToMove();

        std::array<std::array<i64, Width>, 2> accumulators {};

        for (u08 side = 0; side < 2; side++) {
            const u08 perspective = static_cast<u08>(stm ^ side);

            for (s00 i = 0; i < Width; i++) accumulators[side][i] = read(BiasBegin + i);

            for (u08 square = 0; square < 64; square++) {
                const u32 code = position.At(square);

                if (code == Empty) continue;

                const u32 feature = 384 * (code / 6 != perspective) + 64 * (code % 6) + (square ^ 56 * perspective);

                for (s00 i = 0; i < Width; i++) accumulators[side][i] += read(feature * Width + i);
            }
        }

        i64 output = read(OffsetBegin);

        for (u08 side = 0; side < 2    ; side++)
        for (s00 i    = 0; i    < Width; i++   )
            output += std::clamp<i64>(accumulators[side][i], 0, 255) * read(WeightBegin + side * Width + i);

        return static_cast<i32>(output * 400 / (255 * 64));
    }

    template<auto D>
    void Check(Random& random)
    {
        const std::vector<std::byte> bytes = RandomBytes<D>(random);

        const auto network = std::make_unique<Network<D>>();

        const LoadResult result = network->Load(bytes);

        ASSERT_TRUE(result) << result.Message();

        for (const std::string_view fen : Fens)
        for (const u08 stm : { White, Black }) {
            Reference::Position position { fen };

            if (position.SideToMove() != stm) position.Pass();

            EXPECT_EQ(network->Evaluate(position), ReferenceEvaluate<D>(bytes, position))
                << "stm " << s00 { stm } << ": " << fen;
        }
    }

}

TEST(Reference, Perspective)
{
    Random random(0x5245465045525350);

    ReferenceTest::Check<Examples::Starshard>(random);
    ReferenceTest::Check<Examples::Aurora   >(random);
    ReferenceTest::Check<Examples::Flare    >(random);
}

#endif //MANTARAY_RUNTIME_GTESTREFERENCE_H
