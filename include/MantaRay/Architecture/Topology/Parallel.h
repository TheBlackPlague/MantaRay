//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_PARALLEL_H
#define MANTARAY_PARALLEL_H

#include <tuple>

namespace MantaRay
{

    template<typename... T>
    struct Parallel
    {

        using Branches = std::tuple<T...>;

    };

}

#endif
