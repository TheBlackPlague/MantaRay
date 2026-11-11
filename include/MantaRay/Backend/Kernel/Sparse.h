//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_SPARSE_H
#define MANTARAY_BACKEND_KERNEL_SPARSE_H

#include "Nnz.h"
#include "../../Common.h"
#include "../../Lowering/Facts.h"
#include "../../Lowering/Program.h"

namespace MantaRay::Backend::Kernel
{

    template<u32 Inputs, u32 Rows, u32 Stride = Rows, typename A>
    [[clang::always_inline]]
    void Sparse(const u08* input, const i08* weights, A* sums) noexcept
    {
        static_assert(Inputs % 4 == 0, "sparse inputs are scanned in groups of 4 bytes");
        static_assert(Stride >= Rows , "a chunk column holds every row"                );

        constexpr u32 Groups = Inputs / 4;

        Array<u32, Groups> active;

        u32 count = 0;

        for (u32 group = 0; group < Groups; group++) {
            const u08* x = input + group * 4;

            active[count]  = group;
            count         += (x[0] | x[1] | x[2] | x[3]) != 0;
        }

        for (u32 i = 0; i < count; i++) {
            const u08* x   = input   + active[i] * 4;
            const i08* row = weights + active[i] * 4 * Stride;

            for (u32 r = 0; r < Rows; r++)
            for (u32 j = 0; j < 4   ; j++) sums[r] += static_cast<A>(x[j]) * static_cast<A>(row[r * 4 + j]);
        }
    }

    namespace Detail
    {

        template<typename ISA, Lowering::Dot D>
        [[clang::always_inline]]
        ISA::Vector SparseDot(
            const typename ISA::Vector sum,
            const typename ISA::Vector x  ,
            const typename ISA::Vector w
        ) noexcept
        {
            if constexpr (D == Lowering::Dot::Wide) return ISA::DotU08I08Wide(sum, x, w);
            else                                    return ISA::DotU08I08    (sum, x, w);
        }

        template<typename ISA, Lowering::Dot D, s00 Tile, u64 Column>
        [[clang::always_inline]]
        void SparseChunk(
                  Array<typename ISA::Vector, Tile>& sums   ,
            const u08*                               input  ,
            const i08*                               weights,
            const u32                                group
        ) noexcept
        {
            const auto x      = ISA::BroadcastU32(input + group * 4);
            const i08* column = weights + group * Column;

            UNROLL(full)
            for (s00 t = 0; t < Tile; t++) sums[t] = SparseDot<ISA, D>(sums[t], x, ISA::Load(column + t * ISA::Bytes));
        }

        template<typename ISA, s00 Tile, u64 Column>
        [[clang::always_inline]]
        void SparsePair(
                  Array<typename ISA::Vector, Tile>& sums   ,
            const u08*                               input  ,
            const i08*                               weights,
            const u32                                first  ,
            const u32                                next
        ) noexcept
        {
            const auto ones = ISA::BroadcastI16(1);

            const auto a = ISA::BroadcastU32(input + first * 4);
            const auto b = ISA::BroadcastU32(input + next  * 4);

            const i08* left  = weights + first * Column;
            const i08* right = weights + next  * Column;

            UNROLL(full)
            for (s00 t = 0; t < Tile; t++) {
                const auto low  = ISA::MaddubsI16(a, ISA::Load(left  + t * ISA::Bytes));
                const auto high = ISA::MaddubsI16(b, ISA::Load(right + t * ISA::Bytes));

                sums[t] = ISA::AddI32(sums[t], ISA::MaddI16(ISA::AddI16(low, high), ones));
            }
        }

    }

    template<const Lowering::Stage& S, typename ISA>
    [[clang::always_inline]]
    void Sparse(
        const u08*                                 input  ,
        const i08*                                 weights,
              Array<typename ISA::Vector, S.Tile>& sums
    ) noexcept
    {
        constexpr Interval X { .Min = S.Low      , .Max = S.High       };
        constexpr Interval W { .Min = S.WeightLow, .Max = S.WeightHigh };

        constexpr u32  Groups   = S.Width / 4;
        constexpr s00  Tile     = S.Tile;
        constexpr s00  Sets     = S.Sets;
        constexpr u32  Take     = S.Pairs ? 2 : 1;
        constexpr u32  Step     = Sets * Take;
        constexpr u64  Column   = u64 { S.Padded } * 4;
        constexpr bool Positive = S.Test == Lowering::Test::Positive;

        static_assert(
            S.Kind == Lowering::Kind::Dense && S.Form == Lowering::Form::Sparse,
            "Kernel::Sparse runs a sparse Dense stage"
        );
        static_assert(S.Width % 4 == 0 && !S.Wide, "a sparse stage reads 4-byte groups into a modular i32 sum");
        static_assert(Tile * ISA::Bytes == Column, "the tile covers the padded rows of a chunk column");
        static_assert(S.Padded >= S.Rows && Sets >= 1, "padding only adds rows, and a sparse stage has a set");
        static_assert(S.Test != Lowering::Test::None, "a sparse stage names its NNZ test");
        static_assert(!Positive || Lowering::NnzCmpgtSafe(X), "the cmpgt NNZ test needs NnzCmpgtSafe");

        static_assert(
            S.Dot == Lowering::Dot::Dpbusd || S.Dot == Lowering::Dot::Maddubs || S.Dot == Lowering::Dot::Wide,
            "a sparse stage dots u08 by i08"
        );
        static_assert(
            S.Dot != Lowering::Dot::Dpbusd || (ISA::VNNI && Lowering::VnniExact(X, W)),
            "vpdpbusd needs VNNI and VnniExact"
        );
        static_assert(
            S.Dot != Lowering::Dot::Maddubs || Lowering::MaddubsExact(X, W),
            "the saturating u08 by i08 dot needs MaddubsExact"
        );
        static_assert(
            !S.Pairs || (S.Dot == Lowering::Dot::Maddubs && Lowering::MaddubsPairExact(X, W)),
            "two chunks share one i16 sum only under MaddubsPairExact"
        );

        alignas(Alignment) NnzList<Groups> list;

        const u32 count = Nnz<ISA, Groups, Positive>(input, list.data());

        Array<typename ISA::Vector, Sets, Tile> sets;

        sets[0] = sums;

        for (s00 s = 1; s < Sets; s++)
        for (s00 t = 0; t < Tile; t++) sets[s][t] = ISA::Zero();

        u32 i = 0;

        UNROLL(disable)
        for (; i + Step <= count; i += Step) {
            UNROLL(full)
            for (s00 s = 0; s < Sets; s++) {
                const u32 first = i + static_cast<u32>(s) * Take;

                if constexpr (S.Pairs)
                    Detail::SparsePair<ISA, Tile, Column>(sets[s], input, weights, list[first], list[first + 1]);
                else
                    Detail::SparseChunk<ISA, S.Dot, Tile, Column>(sets[s], input, weights, list[first]);
            }
        }

        if constexpr (S.Pairs) {
            UNROLL(disable)
            for (; i + 2 <= count; i += 2)
                Detail::SparsePair<ISA, Tile, Column>(sets[0], input, weights, list[i], list[i + 1]);
        }

        UNROLL(disable)
        for (; i < count; i++) Detail::SparseChunk<ISA, S.Dot, Tile, Column>(sets[0], input, weights, list[i]);

        for (s00 t = 0; t < Tile; t++) {
            sums[t] = sets[0][t];

            for (s00 s = 1; s < Sets; s++) sums[t] = ISA::AddI32(sums[t], sets[s][t]);
        }
    }

}

#endif //MANTARAY_BACKEND_KERNEL_SPARSE_H
