//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
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
