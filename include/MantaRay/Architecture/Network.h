//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_NETWORK_H
#define MANTARAY_NETWORK_H

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
