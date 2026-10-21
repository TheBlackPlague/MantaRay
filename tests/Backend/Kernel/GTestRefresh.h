//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_GTESTREFRESH_H
#define MANTARAY_BACKEND_KERNEL_GTESTREFRESH_H

#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Backend/Kernel/Update.h>
#include <MantaRay/Backend/Target.h>

#include "../ContainerUtil.h"

using namespace MantaRay;

template<typename T, s00 N>
void CheckRefresh(const s00 subs, const s00 adds)
{
    const Array<T, N> base = Ramp<T, N, 30>();

    std::vector<Array<T, N>> rows(subs + adds);

    for (s00 k = 0; k < rows.size(); k++)
    for (s00 i = 0; i < N          ; i++) rows[k][i] = static_cast<T>((k + 1) * (i % 5 + 1));

    std::vector<const T*> subtract;
    std::vector<const T*> add     ;

    for (s00 k = 0; k < subs; k++) subtract.push_back(rows[       k].data());
    for (s00 k = 0; k < adds; k++) add     .push_back(rows[subs + k].data());

    Array<T, N> expected = base;

    for (s00 i = 0; i < N; i++) {
        for (const T* row : subtract) expected[i] = static_cast<T>(expected[i] - row[i]);
        for (const T* row : add     ) expected[i] = static_cast<T>(expected[i] + row[i]);
    }

    Array<T, N> destination;
    Array<T, N> mirror     ;

    Backend::Kernel::Refresh<Backend::Native, N>(base.data(), destination.data(), mirror.data(), subtract, add);

    EXPECT_EQ(destination.Values, expected.Values) << subs << " / " << adds;
    EXPECT_EQ(mirror     .Values, expected.Values) << subs << " / " << adds;
    EXPECT_TRUE(destination.Intact() && mirror.Intact());

    Array<T, N> single;

    Backend::Kernel::Refresh<Backend::Native, N>(base.data(), single.data(), nullptr, subtract, add);

    EXPECT_EQ(single.Values, expected.Values) << subs << " / " << adds;
    EXPECT_TRUE(single.Intact());

    Array<T, N> entry = base;
    Array<T, N> target;

    Backend::Kernel::Refresh<Backend::Native, N>(entry.data(), target.data(), entry.data(), subtract, add);

    EXPECT_EQ(target.Values, expected.Values) << subs << " / " << adds;
    EXPECT_EQ(entry .Values, expected.Values) << subs << " / " << adds;
    EXPECT_TRUE(target.Intact() && entry.Intact());
}

template<typename T, s00 N>
void CheckRefreshCounts()
{
    CheckRefresh<T, N>( 0,  0);
    CheckRefresh<T, N>( 1,  0);
    CheckRefresh<T, N>( 0,  1);
    CheckRefresh<T, N>( 2,  3);
    CheckRefresh<T, N>(31,  1);
    CheckRefresh<T, N>(32, 32);
}

TEST(Refresh, i16_512 ) { CheckRefreshCounts<i16,  512>(); }
TEST(Refresh, i16_768 ) { CheckRefreshCounts<i16,  768>(); }
TEST(Refresh, i16_2048) { CheckRefreshCounts<i16, 2048>(); }
TEST(Refresh, i32_512 ) { CheckRefreshCounts<i32,  512>(); }
TEST(Refresh, i32_2048) { CheckRefreshCounts<i32, 2048>(); }

#endif //MANTARAY_BACKEND_KERNEL_GTESTREFRESH_H
