//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONTAINER_H
#define MANTARAY_CONTAINER_H

#include <array>
#include <cassert>
#include <span>

#include "Constraint.h"

namespace MantaRay
{

    // Short-notation for container types

    template<QuantizedInteger T, s00 N>
    using Array = std::array<T, N>;

    template<s00 Begin, s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    constexpr std::span<const T, Size> Slice(const Array<T, N>& array) requires (Begin <= N && Size <= N - Begin)
    {
        return std::span<const T, Size>(array.data() + Begin, Size);
    }

    template<s00 Begin, s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    constexpr std::span<T, Size> Slice(Array<T, N>& array) requires (Begin <= N && Size <= N - Begin)
    {
        return std::span<T, Size>(array.data() + Begin, Size);
    }

    template<s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    std::span<const T, Size> Slice(const Array<T, N>& array, const s00 begin) requires (Size <= N)
    {
        assert(begin <= N - Size);

        return std::span<const T, Size>(array.data() + begin, Size);
    }

    template<s00 Size, QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    std::span<T, Size> Slice(Array<T, N>& array, const s00 begin) requires (Size <= N)
    {
        assert(begin <= N - Size);

        return std::span<T, Size>(array.data() + begin, Size);
    }

} // MantaRay

#endif //MANTARAY_CONTAINER_H
