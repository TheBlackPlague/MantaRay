//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_FINNYTABLE_H
#define MANTARAY_RUNTIME_FINNYTABLE_H

#include <bit>
#include <cassert>
#include <span>

#include "Network.h"
#include "../Common.h"
#include "../Architecture/FeatureSet.h"
#include "../Backend/Target.h"
#include "../Backend/Kernel/Update.h"

namespace MantaRay
{

    template<typename A, typename ISA = Backend::Native>
    class FinnyTable
    {

        using N = Network<A, ISA>;
        using T = N::Accumulator;
        using F = A::Transformer::Features;

        constexpr static s00 H = N::HiddenSize;

        struct Entry
        {

            alignas(Alignment) Array<T, H> Values;

            Array<u64, 12> Pieces;

        };

        Array<Entry, 2, F::KeyCount> Entries {};

        public:
        void Clear(const N& network) noexcept
        {
            for (Array<Entry, F::KeyCount>& entries : Entries)
            for (Entry&                     entry   : entries) {
                Backend::Kernel::Copy<ISA, H>(network.FeatureBias(), entry.Values);

                entry.Pieces = {};
            }
        }

        template<typename PieceBoards>
        void Refresh(
            const N&           network    ,
            const u08          perspective,
            const u08          key        ,
            const PieceBoards& boards     ,
                  Array<T, H>& destination
        ) noexcept
        {
            assert(perspective < 2 && key < F::KeyCount);

            Entry& entry = Entries[perspective][key];

            Array<const Array<T, H>*, F::MaxActive> subtract;
            Array<const Array<T, H>*, F::MaxActive> add     ;

            s00 subs = 0;
            s00 adds = 0;

            for (u08 color = 0; color < 2; color++)
            for (u08 piece = 0; piece < 6; piece++) {
                const auto row = [&](const u64 board) {
                    const auto square = static_cast<u08>(std::countr_zero(board));

                    return &network.FeatureRow(F::Index(perspective, key, PieceCode(piece, color, square)));
                };

                const u64 cached  = entry.Pieces[color * 6 + piece];
                const u64 current = boards(piece, color);

                u64 kept = cached;

                for (u64 board = cached & ~current; board != 0; board &= board - 1) {
                    assert(subs < F::MaxActive);

                    if (subs == F::MaxActive) break;

                    subtract[subs++] = row(board);

                    kept ^= board & ~(board - 1);
                }

                for (u64 board = current & ~cached; board != 0; board &= board - 1) {
                    assert(adds < F::MaxActive);

                    if (adds == F::MaxActive) break;

                    add[adds++] = row(board);

                    kept |= board & ~(board - 1);
                }

                entry.Pieces[color * 6 + piece] = kept;
            }

            Backend::Kernel::Refresh<ISA, H>(
                entry.Values,
                destination,
                entry.Values,
                std::span(subtract).first(subs),
                std::span(add     ).first(adds)
            );
        }

    };

}

#endif //MANTARAY_RUNTIME_FINNYTABLE_H
