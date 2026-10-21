//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_DOT_H
#define MANTARAY_BACKEND_KERNEL_DOT_H

#include <array>
#include <limits>
#include <type_traits>

#include "Reduce.h"
#include "Update.h"
#include "../../Architecture/Signal.h"
#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    enum class DotMode : u08 { Linear, Lizard, SquareFirst, Widened, PackusDot };

    struct DotKernel
    {

        DotMode Mode = DotMode::Linear;

        s00 Accumulators = 1;

        bool WideReduce = false;

        i16 Low  = std::numeric_limits<i16>::min();
        i16 High = std::numeric_limits<i16>::max();

    };

    namespace Detail
    {

        template<typename ISA, i16 Low, i16 High>
        [[clang::always_inline]]
        ISA::Vector ClampI16(typename ISA::Vector value)
        {
            if constexpr (Low  > std::numeric_limits<i16>::min()) value = ISA::MaxI16(value, ISA::BroadcastI16(Low ));
            if constexpr (High < std::numeric_limits<i16>::max()) value = ISA::MinI16(value, ISA::BroadcastI16(High));

            return value;
        }

        template<typename ISA, DotKernel Plan>
        [[clang::always_inline]]
        ISA::Vector Accumulate(
            const typename ISA::Vector sum   ,
            const typename ISA::Vector input ,
            const typename ISA::Vector weight
        )
        {
            const auto x = ClampI16<ISA, Plan.Low, Plan.High>(input);

            if constexpr (Plan.Mode == DotMode::Linear)
                return ISA::AddI32(sum, ISA::MaddI16(                       x, weight));
            else if constexpr (Plan.Mode == DotMode::Lizard)
                return ISA::AddI32(sum, ISA::MaddI16(ISA::MulLoI16(x, weight),      x));
            else if constexpr (Plan.Mode == DotMode::SquareFirst)
                return ISA::AddI32(sum, ISA::MaddI16(ISA::MulLoI16(x,      x), weight));
            else {
                const auto even = ISA::BroadcastI32(1      );
                const auto odd  = ISA::BroadcastI32(1 << 16);

                const auto low  = ISA::MulLoI32(ISA::MaddI16(x, ISA::MulLoI16(weight, even)), ISA::MaddI16(x, even));
                const auto high = ISA::MulLoI32(ISA::MaddI16(x, ISA::MulLoI16(weight, odd )), ISA::MaddI16(x, odd ));

                return ISA::AddI32(sum, ISA::AddI32(low, high));
            }
        }

        template<typename ISA>
        [[clang::always_inline]]
        ISA::Vector Fetch(const std::array<const i16*, 2>& rows, const s00 slot, const s00 offset)
        {
            return ISA::Load(rows[slot] + offset);
        }

        template<typename ISA, s00 Subs, s00 Adds>
        [[clang::always_inline]]
        ISA::Vector Fetch(const Transition<i16, Subs, Adds>& transition, const s00 slot, const s00 offset)
        {
            return Advance<ISA>(transition, slot, offset);
        }

        template<s00 H, typename W>
        [[clang::always_inline]]
        std::array<const W*, 2> Blocks(const std::array<const i16*, 2>&, const W* weights)
        {
            return { weights, weights + H };
        }

        template<s00 H, typename W, s00 Subs, s00 Adds>
        [[clang::always_inline]]
        std::array<const W*, 2> Blocks(const Transition<i16, Subs, Adds>& transition, const W* weights)
        {
            const s00 stm = transition.SideToMove;

            return { weights + stm * H, weights + (stm ^ 1) * H };
        }

        consteval s00 Block(const s00 vectors, const s00 chains, const s00 target)
        {
            for (s00 block = target; block < vectors; block++)
                if (vectors % block == 0 && 2 * block % chains == 0) return block;

            return vectors;
        }

        template<typename ISA, DotKernel Plan, s00 H, s00 Unroll, typename Input, typename W>
        [[clang::always_inline]]
        i64 Dot(const Input input, const W* weights)
        {
            constexpr bool Packed = Plan.Mode == DotMode::PackusDot;

            constexpr s00  Lanes   = ISA::Bytes / sizeof(i16);
            constexpr s00  Width   = Packed ? ISA::Bytes : Lanes;
            constexpr s00  K       = Plan.Accumulators;
            constexpr s00  Vectors = H / Width;
            constexpr s00  Step    = Unroll == 0 ? Vectors : Block(Vectors, K, Unroll);

            static_assert(std::is_same_v<W, std::conditional_t<Packed, i08, i16>>, "packed dots take i8 weights");
            static_assert(!Packed || ISA::VNNI, "packed dots need exact u8 x i8 dot products");
            static_assert(H % Width == 0 && VectorBytes % ISA::Bytes == 0, "perspective rows must fill whole vectors");
            static_assert(K > 0, "a dot product needs an accumulator");

            const std::array<const W*, 2> blocks = Blocks<H>(input, weights);

            std::array<typename ISA::Vector, K> sums;

            #pragma clang loop unroll(full)
            for (s00 k = 0; k < K; k++) sums[k] = ISA::Zero();

            #pragma clang loop unroll(disable)
            for (s00 v = 0; v < Vectors; v += Step)
            #pragma clang loop unroll(full)
            for (s00 j = 0; j < Step; j++)
            #pragma clang loop unroll(full)
            for (s00 slot = 0; slot < 2; slot++) {
                const s00  i      = (v + j) * Width;
                const s00  k      = (2 * j + slot) % K;
                const auto weight = ISA::Load(blocks[slot] + i);

                if constexpr (Packed) {
                    const auto low  = Fetch<ISA>(input, slot, i        );
                    const auto high = Fetch<ISA>(input, slot, i + Lanes);

                    sums[k] = ISA::DotU8I8(sums[k], ISA::PackUnsignedI16(low, high), weight);
                } else sums[k] = Accumulate<ISA, Plan>(sums[k], Fetch<ISA>(input, slot, i), weight);
            }

            if constexpr (Plan.WideReduce) return ReduceI64<ISA, K>(sums);
            else                           return ReduceI32<ISA, K>(sums);
        }

    }

    constexpr inline s00 FusedUnroll = 4;

    template<typename ISA, DotKernel Plan, s00 H, typename W>
    [[clang::always_inline]]
    i64 PerspectiveDot(const i16* stm, const i16* ntm, const W* weights)
    {
        return Detail::Dot<ISA, Plan, H, 0>(std::array { stm, ntm }, weights);
    }

    template<typename ISA, DotKernel Plan, s00 H, s00 Subs, s00 Adds, typename W>
    [[clang::always_inline]]
    i64 PerspectiveDot(const Transition<i16, Subs, Adds>& transition, const W* weights)
    {
        return Detail::Dot<ISA, Plan, H, FusedUnroll>(transition, weights);
    }

}

#endif //MANTARAY_BACKEND_KERNEL_DOT_H
