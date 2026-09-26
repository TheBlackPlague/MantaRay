//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_SEQUENCE_H
#define MANTARAY_SEQUENCE_H

#include <tuple>

namespace MantaRay
{

    template<typename... T>
    struct Sequence
    {

        using Nodes = std::tuple<T...>;

    };

}

#endif
