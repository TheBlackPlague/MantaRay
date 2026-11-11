//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_REFERENCE_POSITION_H
#define MANTARAY_REFERENCE_POSITION_H

#include <bit>
#include <string_view>

#include "../Common.h"
#include "../FeatureSet/FeatureSet.h"

namespace MantaRay::Reference
{

    class Position
    {

        Array<u64, 2, 6> Boards {};
        Array<u08, 64>   Codes  {};

        u08 Stm   = White;
        u08 Moves = 0;

        public:
        constexpr explicit Position(const std::string_view fen)
        {
            constexpr std::string_view Letters = "pnbrqk";

            Codes.fill(12);

            u08 rank = 7;
            u08 file = 0;
            s00 i    = 0;

            for (; i < fen.size() && fen[i] != ' '; i++) {
                const char c = fen[i];

                if (c == '/') {
                    rank--;
                    file = 0;

                    continue;
                }

                if (c >= '1' && c <= '8') {
                    file += static_cast<u08>(c - '0');

                    continue;
                }

                const u08 side   = c >= 'a' ? Black : White;
                const u08 lower  = side == Black ? c : static_cast<char>(c - 'A' + 'a');
                const u08 piece  = static_cast<u08>(Letters.find(static_cast<char>(lower)));
                const u08 square = static_cast<u08>(rank * 8 + file++);

                Boards[side][piece] |= u64 { 1 } << square;
                Codes [square]       = static_cast<u08>(side * 6 + piece);
            }

            Stm = i + 1 < fen.size() && fen[i + 1] == 'b' ? Black : White;

            s00 fields = 0;

            for (; i < fen.size(); i++) {
                if (fen[i] == ' ') {
                    fields++;

                    continue;
                }

                if (fields == 4) Moves = static_cast<u08>(Moves * 10 + (fen[i] - '0'));
            }
        }

        constexpr void Remove(const u08 side, const Piece piece, const u08 square)
        {
            Boards[side][static_cast<s00>(piece)] &= ~(u64 { 1 } << square);
            Codes [square]                         = 12;
        }

        constexpr void Place(const u08 side, const Piece piece, const u08 square)
        {
            Boards[side][static_cast<s00>(piece)] |= u64 { 1 } << square;
            Codes [square]                         = static_cast<u08>(side * 6 + static_cast<u08>(piece));
        }

        constexpr void Pass() { Stm ^= 1; }

        constexpr u64 Pieces(const u08 side, const Piece piece) const
        { return Boards[side][static_cast<s00>(piece)]; }

        constexpr u64 Occupied() const
        {
            u64 all = 0;

            for (const Array<u64, 6>& side : Boards)
            for (const u64 board : side) all |= board;

            return all;
        }

        constexpr u08 King(const u08 side) const
        { return static_cast<u08>(std::countr_zero(Pieces(side, Piece::King))); }

        constexpr u08 At(const u08 square) const { return Codes[square]; }

        constexpr u08 SideToMove() const { return Stm; }

        constexpr u08 Clock() const { return Moves; }

    };

}

#endif //MANTARAY_REFERENCE_POSITION_H
