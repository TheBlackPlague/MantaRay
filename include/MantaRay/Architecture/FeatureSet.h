//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_FEATURESET_H
#define MANTARAY_ARCHITECTURE_FEATURESET_H

#include <concepts>

#include "../Common/Integral.h"

namespace MantaRay
{

    template<typename F>
    concept FeatureSet = requires (const u08 perspective, const u08 king, const u08 key, const u16 code) {
        { F::InputSize                     } -> std::convertible_to<s00>;
        { F::MaxActive                     } -> std::convertible_to<s00>;
        { F::KeyCount                      } -> std::convertible_to<s00>;

        { F::Key  (perspective,      king) } -> std::same_as<u08>;
        { F::Index(perspective, key, code) } -> std::same_as<u32>;
    };

    constexpr inline u08 KingPiece = 5;

    constexpr u16 PieceCode(const u08 piece, const u08 color, const u08 square)
    {
        return static_cast<u16>(color << 9 | piece << 6 | square);
    }

    constexpr u08  PieceOf(const u16 code) { return static_cast<u08>(code >>  6 & 7); }
    constexpr u08  ColorOf(const u16 code) { return static_cast<u08>(code >>  9 & 1); }
    constexpr u08 SquareOf(const u16 code) { return static_cast<u08>(code &  63    ); }

}

#endif //MANTARAY_ARCHITECTURE_FEATURESET_H
