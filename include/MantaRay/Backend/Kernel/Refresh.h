//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_REFRESH_H
#define MANTARAY_BACKEND_KERNEL_REFRESH_H

#include <span>

#include "Update.h"
#include "../../Common.h"

namespace MantaRay::Backend::Kernel
{

    template<typename ISA, u32 Width, typename T, typename W, typename... D>
    [[clang::always_inline]]
    void Refresh(const T* const bias, const std::span<const W* const> rows, D* const... destinations) noexcept
    { Update<ISA, Width, T, W>(bias, {}, rows, destinations...); }

}

#endif //MANTARAY_BACKEND_KERNEL_REFRESH_H
