//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_GTESTUPDATE_H
#define MANTARAY_BACKEND_KERNEL_GTESTUPDATE_H

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include <MantaRay/Backend/Target.h>
#include <MantaRay/Backend/Kernel/Update.h>

#include "../ContainerUtil.h"

using namespace MantaRay;

template<typename T, s00 N, s00 Subs, s00 Adds>
void CheckUpdate()
{
    const Array<T, N> source = Ramp<T, N, 30>();

    const std::array<Array<T, N>, 2> removed { Generate<T, N, 4>(), Ramp<T, N, 7>() };
    const std::array<Array<T, N>, 2> added   { Ramp<T, N, 11>(), Generate<T, N, 13>() };

    std::array<const T*, Subs> subtract;
    std::array<const T*, Adds> add     ;

    for (s00 i = 0; i < Subs; i++) subtract[i] = removed[i].data();
    for (s00 i = 0; i < Adds; i++) add     [i] = added  [i].data();

    Array<T, N> destination;

    Backend::Kernel::Update<Backend::Native, N, Subs, Adds>(source.data(), destination.data(), subtract, add);

    for (s00 i = 0; i < N; i++) {
        T expected = source[i];

        for (s00 j = 0; j < Subs; j++) expected = static_cast<T>(expected - removed[j][i]);
        for (s00 j = 0; j < Adds; j++) expected = static_cast<T>(expected + added  [j][i]);

        ASSERT_EQ(destination[i], expected) << "index " << i;
    }

    EXPECT_TRUE(destination.Intact());
}

template<s00 Subs, s00 Adds>
void CheckUpdateShape()
{
    CheckUpdate<i16,  512, Subs, Adds>();
    CheckUpdate<i16, 2048, Subs, Adds>();
    CheckUpdate<i32,  512, Subs, Adds>();
    CheckUpdate<i32, 2048, Subs, Adds>();
}

TEST(Update, Add         ) { CheckUpdateShape<0, 1>(); }
TEST(Update, AddAdd      ) { CheckUpdateShape<0, 2>(); }
TEST(Update, Sub         ) { CheckUpdateShape<1, 0>(); }
TEST(Update, SubAdd      ) { CheckUpdateShape<1, 1>(); }
TEST(Update, SubAddAdd   ) { CheckUpdateShape<1, 2>(); }
TEST(Update, SubSub      ) { CheckUpdateShape<2, 0>(); }
TEST(Update, SubSubAdd   ) { CheckUpdateShape<2, 1>(); }
TEST(Update, SubSubAddAdd) { CheckUpdateShape<2, 2>(); }

template<typename T, s00 N>
void CheckUpdateWraps()
{
    constexpr T Max = std::numeric_limits<T>::max();
    constexpr T Min = std::numeric_limits<T>::min();

    Array<T, N> source;
    Array<T, N> row   ;

    for (s00 i = 0; i < N; i++) {
        source[i] = i % 2 == 0 ? Max : Min;
        row   [i] = 1;
    }

    Array<T, N> destination;

    Backend::Kernel::Update<Backend::Native, N, 0, 1>(source.data(), destination.data(), {}, { row.data() });

    for (s00 i = 0; i < N; i++) ASSERT_EQ(destination[i], i % 2 == 0 ? Min : static_cast<T>(Min + 1));

    Backend::Kernel::Update<Backend::Native, N, 1, 0>(destination.data(), destination.data(), { row.data() }, {});

    for (s00 i = 0; i < N; i++) ASSERT_EQ(destination[i], source[i]);
}

TEST(Update, Wraps)
{
    CheckUpdateWraps<i16, 512>();
    CheckUpdateWraps<i32, 512>();
}

#endif //MANTARAY_BACKEND_KERNEL_GTESTUPDATE_H
