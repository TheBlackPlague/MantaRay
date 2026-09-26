//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYCOPY_H
#define MANTARAY_BACKEND_KERNEL_ARRAYCOPY_H

#include <array>
#include <type_traits>

#include "../Native.h"

namespace MantaRay::Backend::Kernel
{

    template<typename T, s00 N>
    [[clang::always_inline]]
    void Copy(const std::array<T, N>& source, std::array<T, N>& destination)
    {
        if constexpr (std::is_integral_v<T>) {
            constexpr s00 Step = Native::NativeLanes<T>;

            s00 i = 0;

            for (; i + Step <= N; i += Step) Native::Store(destination.data() + i, Native::Load(source.data() + i));

            for (; i < N; i++) destination[i] = source[i];
        } else {
            for (s00 i = 0; i < N; i++) Copy(source[i], destination[i]);
        }
    }

}

#endif
