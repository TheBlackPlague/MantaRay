//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BASE_H
#define MANTARAY_BASE_H

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include <hwy/highway.h>

namespace Highway = hwy::HWY_NAMESPACE;

namespace MantaRay
{

    using u08 = uint8_t ;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using i08 =  int8_t ;
    using i16 =  int16_t;
    using i32 =  int32_t;
    using i64 =  int64_t;

    using f32 =  float;
    using f64 = double;

    using usize = size_t;

    using TID = usize;

    template<typename T, usize N, usize... Ns>
    struct IArray { using Base = std::array<typename IArray<T, Ns...>::Base, N>; };

    template<typename T, usize N>
    struct IArray<T, N> { using Base = std::array<T, N>; };

    template<typename T, usize... Ns>
    struct Array : IArray<T, Ns...>::Base {}; // Fixed-size N-dimensional array of type T

    template<typename T>
    using VarArray = std::vector<T>; // Variable-size heap-allocated array of type T

    using OutputStream = std::ostream; // Output stream type

    using InputStream = std::istream; // Input stream type

    using String = std::string; // String type

    using StringStream = std::stringstream; // String stream type

    using OutputStringStream = std::ostringstream; // Output string stream type

    using InputStringStream = std::istringstream; // Input string stream type

    template<usize Begin, usize Size, typename T, usize N>
    [[clang::always_inline]]
    constexpr const Array<T, Size>& Slice(const Array<T, N>& array) requires (Begin + Size <= N)
    { return *reinterpret_cast<Array<T, Size> const*>(array.data() + Begin); }

    template<usize Begin, usize Size, typename T, usize N>
    [[clang::always_inline]]
    constexpr Array<T, Size>& Slice(Array<T, N>& array) requires (Begin + Size <= N)
    { return *reinterpret_cast<Array<T, Size>*>(array.data() + Begin); }

    template<usize Size, typename T, usize N>
    [[clang::always_inline]]
    const Array<T, Size>& Slice(const Array<T, N>& array, const usize begin)
    {
        assert(begin + Size <= N);

        return *reinterpret_cast<Array<T, Size> const*>(array.data() + begin);
    }

    template<usize Size, typename T, usize N>
    [[clang::always_inline]]
    Array<T, Size>& Slice(Array<T, N>& array, const usize begin)
    {
        assert(begin + Size <= N);

        return *reinterpret_cast<Array<T, Size>*>(array.data() + begin);
    }

    template<typename T>
    concept QuantizedInteger = std::is_same_v<T, i08> || std::is_same_v<T, i16> || std::is_same_v<T, i32>;

    template<typename T>
    using LaneExpression = Highway::ScalableTag<T>;

} // MantaRay

#endif //MANTARAY_BASE_H
