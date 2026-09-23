//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_INTEGRAL_H
#define MANTARAY_INTEGRAL_H

namespace MantaRay
{

    // Short-notation for integral types

    using s00 = std::size_t;

    using u64 = uint64_t;
    using i64 =  int64_t;

    using u32 = uint32_t;
    using i32 =  int32_t;

    using u16 = uint16_t;
    using i16 =  int16_t;

    using u08 =  uint8_t;
    using i08 =   int8_t;

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapAdd(const T lhs, const T rhs)
    {
        T result;
        __builtin_add_overflow(lhs, rhs, &result);
        return result;
    }

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapSub(const T lhs, const T rhs)
    {
        T result;
        __builtin_sub_overflow(lhs, rhs, &result);
        return result;
    }

    template<typename T>
    [[clang::always_inline]]
    constexpr T WrapMul(const T lhs, const T rhs)
    {
        T result;
        __builtin_mul_overflow(lhs, rhs, &result);
        return result;
    }

}

#endif //MANTARAY_INTEGRAL_H
