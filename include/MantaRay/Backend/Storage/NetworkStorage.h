//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STORAGE_NETWORKSTORAGE_H
#define MANTARAY_BACKEND_STORAGE_NETWORKSTORAGE_H

#include "LayerStorage.h"

namespace MantaRay::Backend
{

    template<typename Architecture>
    struct NetworkStorage;

    template<typename Q, typename... Nodes>
    struct NetworkStorage<Network<Q, Nodes...>> : Storage<Q, Sequence<Nodes...>>
    {

        using Architecture = Network<Q, Nodes...>;

        static_assert(ValidArchitecture<Architecture>);

    };

}

#endif
