//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_FEATURESET_KINGBUCKETS_H
#define MANTARAY_ARCHITECTURE_FEATURESET_KINGBUCKETS_H

#include <algorithm>

#include "Chess768.h"
#include "../FeatureSet.h"
#include "../../Common.h"

namespace MantaRay
{

    template<Array<u08, 64> Layout>
    struct KingBuckets
    {

        constexpr static s00 KeyCount = std::ranges::max(Layout) + s00 { 1 };

        constexpr static s00 InputSize = Chess768::InputSize * KeyCount;
        constexpr static s00 MaxActive = Chess768::MaxActive           ;

        constexpr static u08 Key(const u08 perspective, const u08 king) { return Layout[king ^ 56 * perspective]; }

        constexpr static u32 Index(const u08 perspective, const u08 key, const u16 code)
        {
            return static_cast<u32>(Chess768::InputSize * key + Chess768::Index(perspective, 0, code));
        }

    };

    template<Array<u08, 32> Layout>
    struct MirroredKingBuckets
    {

        constexpr static s00 Buckets   = std::ranges::max(Layout) + s00 { 1 };
        constexpr static s00 KeyCount  = Buckets * 2;
        constexpr static s00 InputSize = Chess768::InputSize * Buckets;
        constexpr static s00 MaxActive = Chess768::MaxActive;

        static_assert(KeyCount <= 256, "MirroredKingBuckets supports at most 128 buckets");

        constexpr static u08 Key(const u08 perspective, const u08 king)
        {
            const s00 square = king ^ 56 * perspective;

            const s00 file = square % 8;
            const s00 rank = square / 8;

            return static_cast<u08>(Layout[rank * 4 + std::min<s00>(file, 7 - file)] * 2 + (file > 3));
        }

        constexpr static u32 Index(const u08 perspective, const u08 key, const u16 code)
        {
            const u32 bucket = static_cast<u32>(Chess768::InputSize * (key >> 1));
            const u32 mirror = key & 1 ? 7 : 0;

            return bucket + (Chess768::Index(perspective, 0, code) ^ mirror);
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_FEATURESET_KINGBUCKETS_H
