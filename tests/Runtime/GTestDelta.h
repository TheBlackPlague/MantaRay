//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_GTESTDELTA_H
#define MANTARAY_RUNTIME_GTESTDELTA_H

#include <array>
#include <type_traits>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/FeatureSet.h>
#include <MantaRay/Runtime/Delta.h>

using namespace MantaRay;

namespace DeltaTest
{

    constexpr Delta Castle()
    {
        Delta delta;

        delta.Move(5, 0, 4, 6);
        delta.Move(3, 0, 7, 5);

        return delta;
    }

    constexpr Delta Shape(const u08 removals, const u08 additions)
    {
        Delta delta;

        for (u08 i = 0; i < removals ; i++) delta.Remove(i, 1, static_cast<u08>(10 + i));
        for (u08 i = 0; i < additions; i++) delta.Insert   (i, 0, static_cast<u08>(40 + i));

        return delta;
    }

}

TEST(Delta, Empty)
{
    constexpr Delta delta;

    static_assert(delta.RemovedCount() == 0 && delta.InsertedCount() == 0 && delta.Shape() == Delta::ShapeOf(0, 0));
    static_assert(std::is_trivially_copyable_v<Delta>);
    static_assert(sizeof(Delta) == sizeof(u64) && alignof(Delta) == alignof(u64));
}

TEST(Delta, Move)
{
    Delta delta;

    delta.Move(0, 1, 52, 36);

    EXPECT_EQ(delta.RemovedCount(), 1);
    EXPECT_EQ(delta.InsertedCount  (), 1);
    EXPECT_EQ(delta.Removed(0), PieceCode(0, 1, 52));
    EXPECT_EQ(delta.Inserted  (0), PieceCode(0, 1, 36));
}

TEST(Delta, Castle)
{
    constexpr Delta delta = DeltaTest::Castle();

    static_assert(delta.RemovedCount() == 2 && delta.InsertedCount() == 2);

    static_assert(delta.Removed(0) == PieceCode(5, 0, 4) && delta.Removed(1) == PieceCode(3, 0, 7));
    static_assert(delta.Inserted  (0) == PieceCode(5, 0, 6) && delta.Inserted  (1) == PieceCode(3, 0, 5));
}

TEST(Delta, Capture)
{
    Delta delta;

    delta.Remove(4, 1, 59);
    delta.Move  (2, 0, 2, 59);

    EXPECT_EQ(delta.RemovedCount(), 2);
    EXPECT_EQ(delta.InsertedCount  (), 1);
    EXPECT_EQ(delta.Removed(0), PieceCode(4, 1, 59));
    EXPECT_EQ(delta.Removed(1), PieceCode(2, 0,  2));
    EXPECT_EQ(delta.Inserted  (0), PieceCode(2, 0, 59));
}

TEST(Delta, EveryShape)
{
    for (u08 removals  = 0; removals  <= 2; removals ++)
    for (u08 additions = 0; additions <= 2; additions++) {
        const Delta delta = DeltaTest::Shape(removals, additions);

        EXPECT_EQ(delta.RemovedCount(), removals );
        EXPECT_EQ(delta.InsertedCount  (), additions);
        EXPECT_EQ(delta.Shape       (), Delta::ShapeOf(removals, additions));

        for (u08 i = 0; i < removals ; i++) EXPECT_EQ(delta.Removed(i), PieceCode(i, 1, 10 + i));
        for (u08 i = 0; i < additions; i++) EXPECT_EQ(delta.Inserted  (i), PieceCode(i, 0, 40 + i));
    }

    static_assert(DeltaTest::Shape(2, 2).Inserted(1) == PieceCode(1, 0, 41));
    static_assert(DeltaTest::Shape(1, 2).RemovedCount() == 1 && DeltaTest::Shape(0, 2).InsertedCount() == 2);

    std::array<bool, 9> seen {};

    for (u08 removals  = 0; removals  <= 2; removals ++)
    for (u08 additions = 0; additions <= 2; additions++) {
        const i32 shape = Delta::ShapeOf(removals, additions);

        ASSERT_GE(shape, 0);
        ASSERT_LT(shape, 9);
        EXPECT_FALSE(seen[shape]);

        seen[shape] = true;
    }
}

TEST(Delta, ExtremeCodes)
{
    for (u08 high = 0; high < 16; high++) {
        Delta delta;

        constexpr auto Code = [](const bool large) { return large ? PieceCode(5, 1, 63) : PieceCode(0, 0, 0); };

        const auto add    = [&](const bool large) { large ? delta.Insert   (5, 1, 63) : delta.Insert   (0, 0, 0); };
        const auto remove = [&](const bool large) { large ? delta.Remove(5, 1, 63) : delta.Remove(0, 0, 0); };

        remove(high >> 0 & 1);
        remove(high >> 1 & 1);
        add   (high >> 2 & 1);
        add   (high >> 3 & 1);

        EXPECT_EQ(delta.Shape(), Delta::ShapeOf(2, 2));
        EXPECT_EQ(delta.RemovedCount(), 2);
        EXPECT_EQ(delta.InsertedCount  (), 2);
        EXPECT_EQ(delta.Removed(0), Code(high >> 0 & 1));
        EXPECT_EQ(delta.Removed(1), Code(high >> 1 & 1));
        EXPECT_EQ(delta.Inserted  (0), Code(high >> 2 & 1));
        EXPECT_EQ(delta.Inserted  (1), Code(high >> 3 & 1));
    }
}

TEST(DeltaDeathTest, Overflow)
{
#ifdef NDEBUG
    GTEST_SKIP() << "assertions are disabled";
#else
    EXPECT_DEATH({ Delta delta = DeltaTest::Shape(2, 0); delta.Remove(0, 0, 0); }, "RemovedCount\\(\\) < 2");
    EXPECT_DEATH({ Delta delta = DeltaTest::Shape(0, 2); delta.Insert   (0, 0, 0); }, "InsertedCount\\(\\) < 2");
#endif
}

#endif //MANTARAY_RUNTIME_GTESTDELTA_H
