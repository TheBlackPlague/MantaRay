//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
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
