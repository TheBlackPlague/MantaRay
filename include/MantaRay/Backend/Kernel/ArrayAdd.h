//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYADD_H
#define MANTARAY_BACKEND_KERNEL_ARRAYADD_H

#include "../Native.h"

namespace MantaRay::Backend::Kernel
{

    template<typename T, s00 N>
    [[clang::always_inline]]
    void Add(std::array<T, N>& base, const std::array<T, N>& delta)
    {
        constexpr s00 Step = Native::NativeLanes<T>;
        s00 i = 0;

        for (; i + Step <= N; i += Step) Native::Store(
            base.data() + i,
            Native::Add(
                Native::Load( base.data() + i),
                Native::Load(delta.data() + i)
            )
        );

        for (; i < N; i++) base[i] = ISA::Add(base[i], delta[i]);
    }

}

#endif
