//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_GTESTCOPY_H
#define MANTARAY_BACKEND_KERNEL_GTESTCOPY_H

#include <gtest/gtest.h>

#include <MantaRay/Backend/Kernel/Update.h>
#include <MantaRay/Backend/Target.h>

#include "../ContainerUtil.h"

using namespace MantaRay;

template<typename T, s00 N>
void CheckCopy()
{
    const Array<T, N> a = Ramp<T, N, 30>();
          Array<T, N> b ;

    Backend::Kernel::Copy<Backend::Native, N>(a.data(), b.data());

    for (s00 i = 0; i < N; i++) EXPECT_EQ(a[i], b[i]);

    EXPECT_TRUE(b.Intact());
}

TEST(Copy, i16_512 ) { CheckCopy<i16,  512>(); }
TEST(Copy, i16_2048) { CheckCopy<i16, 2048>(); }
TEST(Copy, i32_512 ) { CheckCopy<i32,  512>(); }
TEST(Copy, i32_2048) { CheckCopy<i32, 2048>(); }

#endif //MANTARAY_BACKEND_KERNEL_GTESTCOPY_H
