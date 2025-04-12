//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINER_H
#define MANTARAY_CONTAINER_H

#include <array>

#include "Constraint.h"

namespace MantaRay
{

    // Short-notation for container types

    template<QuantizedInteger T, s00 N>
    using Array = std::array<T, N>;

    template<s00 Begin, s00 End, QuantizedInteger T, s00 N>
    constexpr inline const Array<T, End - Begin>& Slice(const Array<T, N>& array) requires Begin <= End && End <= N
    {
        return *reinterpret_cast<Array<T, End - Begin> const*>(array.data() + Begin);
    }

    template<s00 Begin, s00 End, QuantizedInteger T, s00 N>
    constexpr inline Array<T, End - Begin>& Slice(const Array<T, N>& array) requires Begin <= End && End <= N
    {
        return *reinterpret_cast<Array<T, End - Begin>      *>(array.data() + Begin);
    }

} // MantaRay

#endif //MANTARAY_CONTAINER_H
