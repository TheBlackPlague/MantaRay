//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_FINNYTABLE_H
#define MANTARAY_RUNTIME_FINNYTABLE_H

#include <array>
#include <bit>
#include <cassert>
#include <memory>
#include <span>

#include "Network.h"
#include "../Architecture/FeatureSet.h"
#include "../Backend/Target.h"
#include "../Backend/Kernel/Update.h"
#include "../Common/Alignment.h"
#include "../Common/Integral.h"

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

            alignas(Alignment) std::array<T, H> Values;

            std::array<u64, 12> Pieces;

        };

        std::array<std::array<Entry, F::KeyCount>, 2> Entries {};

        public:
        void Clear(const N& network) noexcept
        {
            for (std::array<Entry, F::KeyCount>& entries : Entries)
            for (Entry&                          entry   : entries) {
                T* values = std::assume_aligned<Alignment>(entry.Values.data());

                Backend::Kernel::Copy<ISA, H>(network.FeatureBias(), values);

                entry.Pieces = {};
            }
        }

        template<typename PieceBoards>
        void Refresh(
            const N&           network    ,
            const u08          perspective,
            const u08          key        ,
            const PieceBoards& boards     ,
                  T*           destination
        ) noexcept
        {
            assert(perspective < 2 && key < F::KeyCount);

            Entry& entry = Entries[perspective][key];

            T* values = std::assume_aligned<Alignment>(entry.Values.data());

            std::array<const T*, F::MaxActive> subtract;
            std::array<const T*, F::MaxActive> add     ;

            s00 subs = 0;
            s00 adds = 0;

            for (u08 color = 0; color < 2; color++)
            for (u08 piece = 0; piece < 6; piece++) {
                const auto row = [&](const u64 board) {
                    const auto square = static_cast<u08>(std::countr_zero(board));

                    return network.FeatureRow(F::Index(perspective, key, PieceCode(piece, color, square)));
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
                values,
                destination,
                values,
                std::span(subtract).first(subs),
                std::span(add     ).first(adds)
            );
        }

    };

}

#endif //MANTARAY_RUNTIME_FINNYTABLE_H
