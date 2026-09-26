//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_NETWORK_H
#define MANTARAY_ARCHITECTURE_NETWORK_H

#include <tuple>

namespace MantaRay
{

    template<typename Q, typename... N>
    struct Network
    {

        using Quantization = Q;

        using Nodes = std::tuple<N...>;

    };

}

#endif
