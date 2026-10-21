//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_PACK_H
#define MANTARAY_BACKEND_KERNEL_PACK_H

#include <array>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    template<typename ISA>
    constexpr std::array<u08, ISA::Bytes> PackPermutation()
    {
        constexpr s00 Half = ISA::Bytes / 2;

        std::array<u08, ISA::Bytes> order;

        for (s00 i = 0; i < ISA::Bytes; i++) order[i] = static_cast<u08>(i / 16 * 8 + i % 8 + (i % 16 < 8 ? 0 : Half));

        return order;
    }

    template<typename ISA, bool Packed>
    constexpr s00 LogicalIndex(const s00 position)
    {
        if constexpr (!Packed) return position;
        else {
            constexpr std::array<u08, ISA::Bytes> Order = PackPermutation<ISA>();

            return position - position % ISA::Bytes + Order[position % ISA::Bytes];
        }
    }

}

#endif //MANTARAY_BACKEND_KERNEL_PACK_H
