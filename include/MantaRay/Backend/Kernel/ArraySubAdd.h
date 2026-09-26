//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYSUBADD_H
#define MANTARAY_BACKEND_KERNEL_ARRAYSUBADD_H

#include "../Native.h"

namespace MantaRay::Backend::Kernel
{

    template<typename T, s00 N>
    [[clang::always_inline]]
    void SubAdd(
              std::array<T, N>& base,
        const std::array<T, N>& sub ,
        const std::array<T, N>& add
    )
    {
        constexpr s00 Step = Native::NativeLanes<T>;

        s00 i = 0;

        for (; i + Step <= N; i += Step) Native::Store(
            base.data() + i,
            Native::Add(
                Native::Sub(
                    Native::Load(base.data() + i),
                    Native::Load( sub.data() + i)
                ),
                Native::Load(add.data() + i)
            )
        );

        for (; i < N; i++) base[i] = ISA::Add(ISA::Sub(base[i], sub[i]), add[i]);
    }

}

#endif
