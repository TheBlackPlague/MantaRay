//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINER_H
#define MANTARAY_CONTAINER_H

#include <array>

namespace MantaRay
{

    // Short-notation for container types

    template<typename T, size_t N>
    using Array = std::array<T, N>;

}

#endif //MANTARAY_CONTAINER_H
