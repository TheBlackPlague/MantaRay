//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_DENSE_H
#define MANTARAY_BACKEND_KERNEL_DENSE_H

#include <array>
#include <type_traits>

#include "Reduce.h"
#include "../../Architecture/Signal.h"
#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    template<typename ISA, s00 In, s00 K, bool Wide, typename T, typename W>
    [[clang::always_inline]]
    i64 DenseDot(const T* input, const W* row)
    {
        constexpr bool Narrow = std::is_same_v<T, u08>;

        static_assert(
            (Narrow && std::is_same_v<W, i08>) || (std::is_same_v<T, i16> && std::is_same_v<W, i16>),
            "dense integer inputs are u8 x i8 or i16 x i16"
        );

        constexpr s00 Lanes = ISA::Bytes / sizeof(T);

        static_assert(In % Lanes == 0 && VectorBytes % ISA::Bytes == 0, "dense rows must fill whole vectors");
        static_assert(K > 0, "a dot product needs an accumulator");

        std::array<typename ISA::Vector, K> sums;

        #pragma clang loop unroll(full)
        for (s00 k = 0; k < K; k++) sums[k] = ISA::Zero();

        #pragma clang loop unroll(full)
        for (s00 v = 0; v < In / Lanes; v++) {
            const auto x = ISA::Load(input + v * Lanes);
            const auto w = ISA::Load(row   + v * Lanes);

            if constexpr (Narrow) sums[v % K] = ISA::DotU8I8(sums[v % K],              x, w) ;
            else                  sums[v % K] = ISA::AddI32 (sums[v % K], ISA::MaddI16(x, w));
        }

        if constexpr (Wide) return ReduceI64<ISA, K>(sums);
        else                return ReduceI32<ISA, K>(sums);
    }

    template<typename ISA, s00 In>
    [[clang::always_inline]]
    f32 DenseDotF(const f32* input, const f32* row)
    {
        #pragma clang fp contract(off)

        constexpr s00 Lanes    = ISA::Bytes / sizeof(f32);
        constexpr s00 Partials = 16 / Lanes;

        static_assert(In % 16 == 0 && 16 % Lanes == 0, "float rows hold whole groups of 16 canonical lanes");

        std::array<typename ISA::VectorF, Partials> partials;

        #pragma clang loop unroll(full)
        for (s00 p = 0; p < Partials; p++) partials[p] = ISA::ZeroF();

        #pragma clang loop unroll(full)
        for (s00 i = 0; i < In; i += 16)
        #pragma clang loop unroll(full)
        for (s00 p = 0; p < Partials; p++) {
            const auto x = ISA::LoadF(input + i + p * Lanes);
            const auto w = ISA::LoadF(row   + i + p * Lanes);

            partials[p] = ISA::AddF(partials[p], ISA::MulF(x, w));
        }

        return ReduceF<ISA, Partials>(partials);
    }

}

#endif //MANTARAY_BACKEND_KERNEL_DENSE_H
