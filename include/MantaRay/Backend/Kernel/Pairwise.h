//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_PAIRWISE_H
#define MANTARAY_BACKEND_KERNEL_PAIRWISE_H

#include <bit>
#include <limits>
#include <type_traits>

#include "Dot.h"
#include "../../Architecture/Signal.h"
#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    struct PairwiseKernel
    {

        s00 Shift = 0;

        i16 Low  = std::numeric_limits<i16>::min();
        i16 High = std::numeric_limits<i16>::max();

        i64 Offset = 0;

    };

    namespace Detail
    {

        template<typename ISA, PairwiseKernel Plan>
        [[clang::always_inline]]
        ISA::Vector Product(const typename ISA::Vector left, const typename ISA::Vector right)
        {
            static_assert(Plan.Shift <= 16, "pairwise shifts above 16 have no i16 kernel");

            const auto a = ClampI16<ISA, Plan.Low, Plan.High>(left );
            const auto b = ClampI16<ISA, Plan.Low, Plan.High>(right);

            constexpr i16 Bias  = std::bit_cast<i16>(static_cast<u16>(         - Plan.Offset));
            constexpr i16 Scale = std::bit_cast<i16>(static_cast<u16>(1 << (16 - Plan.Shift)));

            const auto product = Plan.Shift == 0 ? ISA::MulLoI16(a, b)
                                                 : ISA::MulHiI16(ISA::MulLoI16(a, ISA::BroadcastI16(Scale)), b);

            if constexpr (Plan.Offset == 0) return product;
            else                            return ISA::AddI16(product, ISA::BroadcastI16(Bias));
        }

        template<typename ISA, PairwiseKernel Plan, s00 H>
        [[clang::always_inline]]
        ISA::Vector ProductAt(const i16* stm, const i16* ntm, const s00 position)
        {
            constexpr s00 Half = H / 2;

            const i16* source = position < Half ? stm      : ntm            ;
            const s00  offset = position < Half ? position : position - Half;

            return Product<ISA, Plan>(ISA::Load(source + offset), ISA::Load(source + Half + offset));
        }

    }

    template<typename ISA, PairwiseKernel Plan, s00 H, typename T>
    [[clang::always_inline]]
    void Pairwise(const i16* stm, const i16* ntm, T* output)
    {
        constexpr s00 Lanes = ISA::Bytes / sizeof(i16);

        static_assert(std::is_same_v<T, i16> || std::is_same_v<T, u08>, "pairwise products are stored as i16 or u8");
        static_assert(H / 2 % Lanes == 0 && VectorBytes % ISA::Bytes == 0, "pairwise halves must fill whole vectors");

        if constexpr (std::is_same_v<T, i16>) {
            #pragma clang loop unroll(full)
            for (s00 i = 0; i < H; i += Lanes)
                ISA::Store(output + i, Detail::ProductAt<ISA, Plan, H>(stm, ntm, i));
        } else {
            static_assert(H % (2 * Lanes) == 0, "packed pairwise products must fill whole vectors");

            #pragma clang loop unroll(full)
            for (s00 i = 0; i < H; i += 2 * Lanes) {
                const auto low  = Detail::ProductAt<ISA, Plan, H>(stm, ntm, i        );
                const auto high = Detail::ProductAt<ISA, Plan, H>(stm, ntm, i + Lanes);

                ISA::Store(output + i, ISA::PackUnsignedI16(low, high));
            }
        }
    }

}

#endif //MANTARAY_BACKEND_KERNEL_PAIRWISE_H
