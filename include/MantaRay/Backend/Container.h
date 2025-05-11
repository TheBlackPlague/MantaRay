//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINER_H
#define MANTARAY_CONTAINER_H

#include <array>
#include <cassert>

#include "Constraint.h"

namespace MantaRay
{

    // Short-notation for container types

    template<QuantizedInteger T, s00 N>
    using Array = std::array<T, N>;

    template<s00 Begin, s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    constexpr const Array<T, Size>& Slice(const Array<T, N>& array) requires (Begin + Size <= N)
    {
        return *reinterpret_cast<Array<T, Size> const*>(array.data() + Begin);
    }

    template<s00 Begin, s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    constexpr Array<T, Size>& Slice(Array<T, N>& array) requires (Begin + Size <= N)
    {
        return *reinterpret_cast<Array<T, Size>      *>(array.data() + Begin);
    }

    template<s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    const Array<T, Size>& Slice(const Array<T, N>& array, const s00 begin)
    {
        assert(begin + Size <= N);

        return *reinterpret_cast<Array<T, Size> const*>(array.data() + begin);
    }

    template<s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    Array<T, Size>& Slice(Array<T, N>& array, const s00 begin)
    {
        assert(begin + Size <= N);

        return *reinterpret_cast<Array<T, Size>      *>(array.data() + begin);
    }

} // MantaRay

#endif //MANTARAY_CONTAINER_H
