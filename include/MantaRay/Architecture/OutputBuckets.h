//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_OUTPUTBUCKETS_H
#define MANTARAY_ARCHITECTURE_OUTPUTBUCKETS_H

#include <concepts>

#include "../Common.h"

namespace MantaRay
{

    template<typename B>
    concept OutputBuckets = requires (const u08 pieces) {
        { B::Count      } -> std::convertible_to<s00>;
        { B::Of(pieces) } -> std::same_as       <s00>;
    };

    template<s00 N>
    struct MaterialCount
    {

        static_assert(N > 0, "MaterialCount requires at least one bucket");

        constexpr static s00 Count = N;

        constexpr static s00 Of(const u08 pieces) { return N == 1 ? 0 : (pieces - s00 { 2 }) / ((32 + N - 1) / N); }

    };

}

#endif //MANTARAY_ARCHITECTURE_OUTPUTBUCKETS_H
