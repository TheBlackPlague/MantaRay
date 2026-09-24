//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_NETWORKSTORAGE_H
#define MANTARAY_BACKEND_NETWORKSTORAGE_H

#include "LayerStorage.h"

namespace MantaRay::Backend
{

    template<typename A>
    struct NetworkStorage;

    template<typename Q, typename... N>
    struct NetworkStorage<Network<Q, N...>> : Storage<Q, Sequence<N...>>
    {

        using Architecture = Network<Q, N...>;

        static_assert(ValidArchitecture<Architecture>, "Invalid network architecture or quantization policy.");

    };

}

#endif
