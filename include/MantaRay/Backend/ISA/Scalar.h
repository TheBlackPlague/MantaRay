//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_SCALAR_H
#define MANTARAY_BACKEND_ISA_SCALAR_H

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::ISA
{

    class Scalar
    {

        template<typename T>
        using Lanes [[gnu::vector_size(16)]] = T;

        template<typename T>
        constexpr static s00 Count = 16 / sizeof(T);

        template<typename T, typename V>
        [[clang::always_inline]]
        static V Fill(const T value)
        {
            Lanes<T> lanes;

            for (s00 i = 0; i < Count<T>; i++) lanes[i] = value;

            return std::bit_cast<V>(lanes);
        }

        template<typename T, typename V, typename F>
        [[clang::always_inline]]
        static V Map(const V left, const V right, const F function)
        {
            const auto a = std::bit_cast<Lanes<T>>(left );
            const auto b = std::bit_cast<Lanes<T>>(right);

            Lanes<T> result;

            for (s00 i = 0; i < Count<T>; i++) result[i] = function(a[i], b[i]);

            return std::bit_cast<V>(result);
        }

        [[clang::always_inline]]
        static i32 SaturateI16(const i32 value) { return std::clamp(value, -32768, 32767); }

        public:
        using Vector  = Lanes<u08>;
        using VectorF = Lanes<f32>;

        constexpr static s00  Bytes     = 16;
        constexpr static s00  Registers = 16;
        constexpr static s00  DotChains =  4;
        constexpr static bool VNNI      = false;

        [[clang::always_inline]]
        static Vector Load(const void* address)
        {
            Vector value;

            std::memcpy(&value, std::assume_aligned<Bytes>(static_cast<const u08*>(address)), Bytes);

            return value;
        }

        [[clang::always_inline]]
        static void Store(void* address, const Vector value)
        { std::memcpy(std::assume_aligned<Bytes>(static_cast<u08*>(address)), &value, Bytes); }

        [[clang::always_inline]]
        static Vector Zero() { return Vector {}; }

        [[clang::always_inline]]
        static Vector BroadcastI16(const i16 value) { return Fill<i16, Vector>(value); }

        [[clang::always_inline]]
        static Vector BroadcastI32(const i32 value) { return Fill<i32, Vector>(value); }

        [[clang::always_inline]]
        static Vector AddI16(const Vector left, const Vector right)
        { return Map<i16>(left, right, [](const i16 a, const i16 b) { return WrapAdd(a, b); }); }

        [[clang::always_inline]]
        static Vector SubI16(const Vector left, const Vector right)
        { return Map<i16>(left, right, [](const i16 a, const i16 b) { return WrapSub(a, b); }); }

        [[clang::always_inline]]
        static Vector MinI16(const Vector left, const Vector right)
        { return Map<i16>(left, right, [](const i16 a, const i16 b) { return std::min(a, b); }); }

        [[clang::always_inline]]
        static Vector MaxI16(const Vector left, const Vector right)
        { return Map<i16>(left, right, [](const i16 a, const i16 b) { return std::max(a, b); }); }

        [[clang::always_inline]]
        static Vector MulLoI16(const Vector left, const Vector right)
        { return Map<i16>(left, right, [](const i16 a, const i16 b) { return WrapMul(a, b); }); }

        [[clang::always_inline]]
        static Vector MulHiI16(const Vector left, const Vector right)
        {
            const auto high = [](const i16 a, const i16 b) { return static_cast<i16>(i32 { a } * b >> 16); };

            return Map<i16>(left, right, high);
        }

        [[clang::always_inline]]
        static Vector AddI32(const Vector left, const Vector right)
        { return std::bit_cast<Vector>(std::bit_cast<Lanes<u32>>(left) + std::bit_cast<Lanes<u32>>(right)); }

        [[clang::always_inline]]
        static Vector SubI32(const Vector left, const Vector right)
        { return Map<i32>(left, right, [](const i32 a, const i32 b) { return WrapSub(a, b); }); }

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right)
        { return Map<i32>(left, right, [](const i32 a, const i32 b) { return WrapMul(a, b); }); }

        [[clang::always_inline]]
        static Vector MaddI16(const Vector left, const Vector right)
        {
            using Wide [[gnu::vector_size(32)]] = i32;
            using Bits [[gnu::vector_size(16)]] = u32;

            const auto a = __builtin_convertvector(std::bit_cast<Lanes<i16>>(left ), Wide);
            const auto b = __builtin_convertvector(std::bit_cast<Lanes<i16>>(right), Wide);

            const auto products = a * b;

            const auto even = std::bit_cast<Bits>(__builtin_shufflevector(products, products, 0, 2, 4, 6));
            const auto odd  = std::bit_cast<Bits>(__builtin_shufflevector(products, products, 1, 3, 5, 7));

            return std::bit_cast<Vector>(even + odd);
        }

        [[clang::always_inline]]
        static i32 ReduceI32(const Vector value)
        {
            const auto lanes = std::bit_cast<Lanes<i32>>(value);

            i32 sum = 0;

            for (s00 i = 0; i < Count<i32>; i++) sum = WrapAdd(sum, lanes[i]);

            return sum;
        }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value)
        {
            const auto lanes = std::bit_cast<Lanes<i32>>(value);

            i64 sum = 0;

            for (s00 i = 0; i < Count<i32>; i++) sum += lanes[i];

            return sum;
        }

        [[clang::always_inline]]
        static Vector PackUnsignedI16(const Vector left, const Vector right)
        {
            const auto a = std::bit_cast<Lanes<i16>>(left );
            const auto b = std::bit_cast<Lanes<i16>>(right);

            Vector result;

            for (s00 i = 0; i < Count<i16>; i++) {
                result[i             ] = static_cast<u08>(std::clamp<i16>(a[i], 0, 255));
                result[i + Count<i16>] = static_cast<u08>(std::clamp<i16>(b[i], 0, 255));
            }

            return result;
        }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            const auto s = std::bit_cast<Lanes<i32>>(sum);
            const auto b = std::bit_cast<Lanes<i08>>(i8 );

            Lanes<i32> result;

            for (s00 i = 0; i < Count<i32>; i++) {
                const i32 low  = SaturateI16(u8[4 * i    ] * b[4 * i    ] + u8[4 * i + 1] * b[4 * i + 1]);
                const i32 high = SaturateI16(u8[4 * i + 2] * b[4 * i + 2] + u8[4 * i + 3] * b[4 * i + 3]);

                result[i] = WrapAdd(s[i], low + high);
            }

            return std::bit_cast<Vector>(result);
        }

        [[clang::always_inline]]
        static VectorF LoadF(const f32* address)
        {
            VectorF value;

            std::memcpy(&value, std::assume_aligned<Bytes>(address), Bytes);

            return value;
        }

        [[clang::always_inline]]
        static VectorF ZeroF() { return VectorF {}; }

        [[clang::always_inline]]
        static VectorF AddF(const VectorF left, const VectorF right)
        {
            #pragma clang fp contract(off)

            VectorF result;

            for (s00 i = 0; i < Count<f32>; i++) result[i] = left[i] + right[i];

            return result;
        }

        [[clang::always_inline]]
        static VectorF MulF(const VectorF left, const VectorF right)
        {
            #pragma clang fp contract(off)

            VectorF result;

            for (s00 i = 0; i < Count<f32>; i++) result[i] = left[i] * right[i];

            return result;
        }

        [[clang::always_inline]]
        static f32 ReduceF(VectorF value)
        {
            #pragma clang fp contract(off)

            for (s00 width = Count<f32>; width > 1        ; width /= 2)
            for (s00 i     = 0         ; i     < width / 2; i++) value[i] = value[i] + value[i + width / 2];

            return value[0];
        }

    };

}

#endif //MANTARAY_BACKEND_ISA_SCALAR_H
