//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_GTESTFEATURESET_H
#define MANTARAY_ARCHITECTURE_GTESTFEATURESET_H

#include <array>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Architecture/FeatureSet/Chess768.h>
#include <MantaRay/Architecture/FeatureSet/KingBuckets.h>
#include <MantaRay/Architecture/OutputBuckets.h>

#include "Examples.h"

using namespace MantaRay;

namespace FeatureSetTest
{

    constexpr u32 Direct(const u08 perspective, const u08 piece, const u08 color, const u08 square)
    {
        return 384 * (color != perspective) + 64 * piece + (perspective ? square ^ 56 : square);
    }

    constexpr std::array<u08, 64> Quarters = [] {
        std::array<u08, 64> layout {};

        for (s00 square = 0; square < 64; square++) layout[square] = static_cast<u08>(square % 8 / 2 + square / 32);

        return layout;
    }();

}

TEST(FeatureSet, PieceCode)
{
    static_assert(PieceCode(0, 0,  0) ==    0);
    static_assert(PieceCode(5, 0,  4) ==  324);
    static_assert(PieceCode(0, 1, 48) ==  560);
    static_assert(PieceCode(5, 1, 63) ==  895);
    static_assert(KingPiece == 5);

    static_assert(PieceOf(324) == 5 && ColorOf(324) == 0 && SquareOf(324) ==  4);
    static_assert(PieceOf(560) == 0 && ColorOf(560) == 1 && SquareOf(560) == 48);
    static_assert(PieceOf(895) == 5 && ColorOf(895) == 1 && SquareOf(895) == 63);

    for (u08 color  = 0; color  <  2; color ++)
    for (u08 piece  = 0; piece  <  6; piece ++)
    for (u08 square = 0; square < 64; square++) {
        const u16 code = PieceCode(piece, color, square);

        EXPECT_EQ(PieceOf (code), piece ) << code;
        EXPECT_EQ(ColorOf (code), color ) << code;
        EXPECT_EQ(SquareOf(code), square) << code;
    }

    static_assert(FeatureSet<Chess768> && FeatureSet<KingBuckets<FeatureSetTest::Quarters>>);
    static_assert(FeatureSet<MirroredKingBuckets<Examples::Layout4>>);
}

TEST(FeatureSet, Chess768)
{
    static_assert(Chess768::InputSize == 768 && Chess768::MaxActive == 32 && Chess768::KeyCount == 1);

    for (u08 perspective = 0; perspective <  2; perspective++)
    for (u08 color       = 0; color       <  2; color      ++)
    for (u08 piece       = 0; piece       <  6; piece      ++)
    for (u08 square      = 0; square      < 64; square     ++) {
        const u16 code = PieceCode(piece, color, square);

        EXPECT_EQ(Chess768::Key(perspective, square), 0);
        EXPECT_EQ(Chess768::Index(perspective, 0, code), FeatureSetTest::Direct(perspective, piece, color, square))
            << "perspective " << s00 { perspective } << " code " << code;
    }
}

TEST(FeatureSet, KingBuckets)
{
    using F = KingBuckets<FeatureSetTest::Quarters>;

    static_assert(F::KeyCount == 5 && F::InputSize == 768 * 5 && F::MaxActive == 32);

    for (u08 perspective = 0; perspective <  2; perspective++)
    for (u08 king        = 0; king        < 64; king       ++) {
        const u08 key    = F::Key(perspective, king);
        const u08 bucket = FeatureSetTest::Quarters[perspective ? king ^ 56 : king];

        ASSERT_EQ(key, bucket);

        for (u08 color  = 0; color  <  2; color ++)
        for (u08 piece  = 0; piece  <  6; piece ++)
        for (u08 square = 0; square < 64; square++) {
            const u32 expected = 768 * bucket + FeatureSetTest::Direct(perspective, piece, color, square);

            EXPECT_EQ(F::Index(perspective, key, PieceCode(piece, color, square)), expected);
        }
    }
}

TEST(FeatureSet, MirroredKingBuckets)
{
    using F = MirroredKingBuckets<Examples::Layout4>;

    static_assert(F::KeyCount == 8 && F::InputSize == 768 * 4 && F::MaxActive == 32);

    EXPECT_EQ(F::Key(0,  0), 0);
    EXPECT_EQ(F::Key(0,  7), 1);
    EXPECT_EQ(F::Key(0,  3), 2);
    EXPECT_EQ(F::Key(0,  4), 3);
    EXPECT_EQ(F::Key(0,  8), 4);
    EXPECT_EQ(F::Key(0, 12), 5);
    EXPECT_EQ(F::Key(0, 60), 7);
    EXPECT_EQ(F::Key(1, 59), 2);
    EXPECT_EQ(F::Key(1, 60), 3);
    EXPECT_EQ(F::Key(1,  4), 7);

    for (u08 perspective = 0; perspective <  2; perspective++)
    for (u08 king        = 0; king        < 64; king       ++) {
        const u08 key    = F::Key(perspective, king);
        const u08 mirror = king % 8 > 3;
        const u08 rank   = (perspective ? king ^ 56 : king) / 8;

        EXPECT_EQ(key >> 1, Examples::Layout4[rank * 4 + (mirror ? 7 - king % 8 : king % 8)]);
        EXPECT_EQ(key  & 1, mirror);

        for (u08 color  = 0; color  <  2; color ++)
        for (u08 piece  = 0; piece  <  6; piece ++)
        for (u08 square = 0; square < 64; square++) {
            const u08 seen     = mirror ? square ^ 7 : square;
            const u32 expected = 768 * (key >> 1) + FeatureSetTest::Direct(perspective, piece, color, seen);

            EXPECT_EQ(F::Index(perspective, key, PieceCode(piece, color, square)), expected);
        }
    }
}

TEST(OutputBuckets, MaterialCount)
{
    static_assert(OutputBuckets<MaterialCount<8>> && MaterialCount<8>::Count == 8);

    static_assert(MaterialCount<8>::Of(32) == 7);
    static_assert(MaterialCount<8>::Of( 2) == 0);
    static_assert(MaterialCount<8>::Of(17) == 3);
    static_assert(MaterialCount<1>::Of(32) == 0);
    static_assert(MaterialCount<3>::Of(32) == 2);

    for (u08 pieces = 2; pieces <= 32; pieces++) {
        EXPECT_EQ(MaterialCount<8>::Of(pieces), static_cast<s00>((pieces - 2) / 4));
        EXPECT_EQ(MaterialCount<4>::Of(pieces), static_cast<s00>((pieces - 2) / 8));
        EXPECT_EQ(MaterialCount<1>::Of(pieces), s00 { 0 });
    }
}

#endif //MANTARAY_ARCHITECTURE_GTESTFEATURESET_H
