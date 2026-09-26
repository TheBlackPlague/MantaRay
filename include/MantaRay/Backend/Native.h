//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_NATIVE_H
#define MANTARAY_BACKEND_NATIVE_H

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <type_traits>

#include "ISA/Portable.h"

#if !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX512BW__)
#include "ISA/AVX512.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX512F__)
#include "ISA/AVX512F.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX2__)
#include "ISA/AVX2.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX__)
#include "ISA/AVX.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__SSE4_1__)
#include "ISA/SSE41.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__SSE2__)
#include "ISA/SSE2.h"
#elif !defined(MANTARAY_FORCE_SCALAR) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include "ISA/NEON.h"
#endif

namespace MantaRay::Backend::Native
{

#if !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX512BW__)
    template<typename T> using Policy = ISA::AVX512<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX512F__)
    template<typename T> using Policy = ISA::AVX512F<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX2__)
    template<typename T> using Policy = ISA::AVX2<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__AVX__)
    template<typename T> using Policy = ISA::AVX<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__SSE4_1__)
    template<typename T> using Policy = ISA::SSE41<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && defined(__SSE2__)
    template<typename T> using Policy = ISA::SSE2<T>;
#elif !defined(MANTARAY_FORCE_SCALAR) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    template<typename T> using Policy = ISA::NEON<T>;
#else
    template<typename T> using Policy = ISA::Portable<T>;
#endif

    template<typename T>
    constexpr inline s00 NativeLanes = Policy<T>::Bytes / sizeof(T);

    template<typename T>
    constexpr inline s00 DotAccumulators = [] {
        if constexpr (requires { Policy<T>::DotAccumulators; }) return Policy<T>::DotAccumulators;
        else return s00 { 1 };
    }();

    template<typename T, s00 Lanes = NativeLanes<T>>
    struct Register
    {

        static_assert(std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) <= 4);
        static_assert(Lanes > 0);

        constexpr static bool IsNative = NativeLanes<T> > 1 && Lanes == NativeLanes<T>;

        using Instructions = Policy<T>;

        using Storage = std::conditional_t<IsNative, typename Instructions::Vector, std::array<T, Lanes>>;

        Storage Value;

    };

    template<typename T, s00 Lanes = NativeLanes<T>>
    [[clang::always_inline]]
    Register<T, Lanes> Load(const T* values)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Load(values) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Policy<T>::Store(result.Value.data() + i, Policy<T>::Load(values + i));
            }

            for (; i < Lanes; i++) result.Value[i] = values[i];

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    void Store(T* values, const Register<T, Lanes> value)
    {
        if constexpr (Register<T, Lanes>::IsNative) Policy<T>::Store(values, value.Value);
        else {
            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Policy<T>::Store(values + i, Policy<T>::Load(value.Value.data() + i));
            }

            for (; i < Lanes; i++) values[i] = value.Value[i];
        }
    }

    template<typename T, s00 Lanes = NativeLanes<T>>
    [[clang::always_inline]]
    Register<T, Lanes> LoadAligned(const T* values)
    {
        return Load<T, Lanes>(std::assume_aligned<Policy<T>::Bytes>(values));
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    void StoreAligned(T* values, const Register<T, Lanes> value)
    {
        Store(std::assume_aligned<Policy<T>::Bytes>(values), value);
    }

    template<typename T, s00 Lanes = NativeLanes<T>>
    [[clang::always_inline]]
    Register<T, Lanes> Broadcast(const T value)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Broadcast(value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Policy<T>::Store(result.Value.data() + i, Policy<T>::Broadcast(value));
            }

            for (; i < Lanes; i++) result.Value[i] = value;

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> Add(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Add(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, Add(Load(left.Value.data() + i), Load(right.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = ISA::Add(left.Value[i], right.Value[i]);

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> Sub(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Sub(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, Sub(Load(left.Value.data() + i), Load(right.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = ISA::Sub(left.Value[i], right.Value[i]);

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> Min(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Min(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, Min(Load(left.Value.data() + i), Load(right.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = std::min(left.Value[i], right.Value[i]);

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> Max(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::Max(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, Max(Load(left.Value.data() + i), Load(right.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = std::max(left.Value[i], right.Value[i]);
            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> MulLo(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::MulLo(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, MulLo(Load(left.Value.data() + i), Load(right.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = ISA::Mul(left.Value[i], right.Value[i]);

            return result;
        }
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> MulHighUnsigned(const Register<T, Lanes> left, const Register<T, Lanes> right)
    {
        static_assert(sizeof(T) == 2);

        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::MulHighUnsigned(left.Value, right.Value) };
        else {
            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>) Store(
                    result.Value.data() + i,
                    MulHighUnsigned(Load(left.Value.data() + i), Load(right.Value.data() + i))
                );
            }

            for (; i < Lanes; i++) result.Value[i] = static_cast<T>(
                (static_cast<u32>(static_cast<u16>(left.Value[i])) * static_cast<u16>(right.Value[i])) >> 16
            );

            return result;
        }
    }

    template<s00 Bits, typename T, s00 Lanes>
    [[clang::always_inline]]
    Register<T, Lanes> ShiftRight(const Register<T, Lanes> value)
    {
        static_assert(Bits < sizeof(T) * 8);

        if constexpr (Register<T, Lanes>::IsNative) return { Policy<T>::template ShiftRight<Bits>(value.Value) };
        else {
            using U = std::make_unsigned_t<T>;

            Register<T, Lanes> result;

            s00 i = 0;
            if constexpr (NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>)
                    Store(result.Value.data() + i, ShiftRight<Bits>(Load(value.Value.data() + i)));
            }

            for (; i < Lanes; i++) result.Value[i] = static_cast<T>(static_cast<U>(value.Value[i]) >> Bits);

            return result;
        }
    }

    template<s00 Lanes>
    [[clang::always_inline]]
    Register<i32, Lanes / 2> MultiplyAddPairs(const Register<i16, Lanes> left, const Register<i16, Lanes> right)
    {
        static_assert(Lanes % 2 == 0);

        using Instructions = Register<i16, Lanes>::Instructions;

        if constexpr (Register<i16, Lanes>::IsNative)
            return { Instructions::MultiplyAddPairs(left.Value, right.Value) };
        else {
            Register<i32, Lanes / 2> result;

            s00 i = 0;
            if constexpr (NativeLanes<i16> > 1) {
                for (; i + NativeLanes<i16> <= Lanes; i += NativeLanes<i16>) Store(
                    result.Value.data() + i / 2,
                    MultiplyAddPairs(Load(left.Value.data() + i), Load(right.Value.data() + i))
                );
            }

            for (; i < Lanes; i += 2) result.Value[i / 2] = ISA::Add(
                static_cast<i32>(left.Value[i    ]) * right.Value[i    ],
                static_cast<i32>(left.Value[i + 1]) * right.Value[i + 1]
            );

            return result;
        }
    }

    template<s00 Lanes>
    [[clang::always_inline]]
    Register<i32, Lanes / 2> AccumulateDot(
        const Register<i32, Lanes / 2>   sum,
        const Register<i16, Lanes    >  left,
        const Register<i16, Lanes    > right
    )
    {
        static_assert(Lanes % 2 == 0);

        using Instructions = Register<i16, Lanes>::Instructions;

        if constexpr (Register<i16, Lanes>::IsNative && requires {
            Instructions::AccumulateDot(sum.Value, left.Value, right.Value);
        }) return { Instructions::AccumulateDot(sum.Value, left.Value, right.Value) };

        else return Add(sum, MultiplyAddPairs(left, right));
    }

    template<typename T, s00 Lanes>
    [[clang::always_inline]]
    T Sum(const Register<T, Lanes> value)
    {
        if constexpr (Register<T, Lanes>::IsNative && std::is_same_v<T, i32>) return Policy<T>::Sum(value.Value);
        else {
            std::array<T, Lanes> values;
            Store(values.data(), value);

            T result = 0;

            s00 i = 0;
            if constexpr (!Register<T, Lanes>::IsNative && NativeLanes<T> > 1) {
                for (; i + NativeLanes<T> <= Lanes; i += NativeLanes<T>) result = ISA::Add(
                    result, Sum(Load(values.data() + i))
                );
            }

            for (; i < Lanes; i++) result = ISA::Add(result, values[i]);

            return result;
        }
    }

}

#endif
