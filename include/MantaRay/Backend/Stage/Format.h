//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STAGE_FORMAT_H
#define MANTARAY_BACKEND_STAGE_FORMAT_H

#include <concepts>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "../Facts.h"
#include "../../Common.h"
#include "../../Architecture/Output.h"
#include "../../Architecture/OutputBuckets.h"
#include "../../Architecture/Signal.h"
#include "../../Architecture/Activation/ClippedReLU.h"
#include "../../Architecture/Activation/ReLU.h"
#include "../../Architecture/Activation/SquaredClippedReLU.h"
#include "../../Architecture/Activation/SquaredReLU.h"
#include "../../Architecture/Topology/Pairwise.h"
#include "../../Architecture/Topology/Select.h"
#include "../../Architecture/Transform/Affine.h"
#include "../../IO/Loader.h"

namespace MantaRay::Backend::Stage
{

    template<typename A, s00 I>
    using LayerAt = ItemAt<typename A::Layers, I>;

    template<typename L>
    constexpr inline bool IsAffine = false;

    template<s00 In, s00 Out, typename W, typename B, typename S>
    constexpr inline bool IsAffine<Affine<In, Out, W, B, S>> = true;

    template<typename L>
    constexpr inline bool IsPairwise = false;

    template<Layer A, s00 K>
    constexpr inline bool IsPairwise<Pairwise<A, K>> = true;

    template<typename L>
    constexpr inline bool IsSelect = false;

    template<OutputBuckets B>
    constexpr inline bool IsSelect<Select<B>> = true;

    template<typename L>
    constexpr inline bool IsOutput = false;

    template<i64 S>
    constexpr inline bool IsOutput<Output<S>> = true;

    template<typename L>
    concept Elementwise = requires (const i64 value, const Scale& scale, const f32 real) {
        { L::Apply     (value, scale) } -> std::same_as<i64>;
        { L::ApplyFloat(real        ) } -> std::same_as<f32>;
    };

    template<typename L>
    struct Rectifier
    {

        constexpr static bool Known = false;

    };

    template<>
    struct Rectifier<ReLU>
    {

        constexpr static bool             Known   = true  ;
        constexpr static bool             Squared = false ;
        constexpr static std::string_view Name    = "ReLU";

        constexpr static Interval Bounds(const Scale&) { return { .Min = 0, .Max = std::numeric_limits<i64>::max() }; }

    };

    template<i32 Min, i32 Max>
    struct Rectifier<ClippedReLU<Min, Max>>
    {

        constexpr static bool             Known   = true   ;
        constexpr static bool             Squared = false  ;
        constexpr static std::string_view Name    = "CReLU";

        constexpr static Interval Bounds(const Scale& scale) { return ClippedReLU<Min, Max>::Bounds(scale); }

    };

    template<i32 Min, i32 Max>
    struct Rectifier<SquaredClippedReLU<Min, Max>>
    {

        constexpr static bool             Known   = true    ;
        constexpr static bool             Squared = true    ;
        constexpr static std::string_view Name    = "SCReLU";

        constexpr static Interval Bounds(const Scale& scale) { return ClippedReLU<Min, Max>::Bounds(scale); }

    };

    template<>
    struct Rectifier<SquaredReLU>
    {

        constexpr static bool             Known   = true     ;
        constexpr static bool             Squared = true     ;
        constexpr static std::string_view Name    = "SqrReLU";

        constexpr static Interval Bounds(const Scale&) { return { .Min = 0, .Max = std::numeric_limits<i64>::max() }; }

    };

    template<typename L>
    consteval std::string_view ShortName()
    {
        if constexpr (Rectifier<L>::Known) return Rectifier<L>::Name;

        return L::Name;
    }

    enum class Encoding : u08 { I16, ZeroPoint, U08, I32, I64, F32 };

    struct Format
    {

        Encoding Encoding = Encoding::I16;
        i64      Offset   = 0;

    };

    template<Format F>
    using Element = At<static_cast<s00>(F.Encoding), i16, i16, u08, i32, i64, f32>;

    constexpr std::string_view EncodingName(const Encoding encoding)
    {
        constexpr Array<std::string_view, 6> Names { "I16", "ZeroPoint", "U8", "I32", "I64", "Float" };

        return Names[static_cast<s00>(encoding)];
    }

    template<typename A, s00 I>
    consteval Format FormatOf()
    {
        using L = LayerAt<A, I>;

        constexpr Signal Input = A::Signals[I];

        const Interval range = Input.Range;

        if (Input.Domain == Domain::Float) return { .Encoding = Encoding::F32 };

        if constexpr (IsAffine<L>) {
            const Interval weights = L::WeightSpec::Bounds;

            if (MaddubsExact(range, weights) && Fits<i32>(Repeat(range * weights, L::InputSize)))
                return { .Encoding = Encoding::U08 };

            if (!Fits<i16>(range) && OffsetI16(range) && L::InputSize % (VectorBytes / sizeof(i16)) == 0)
                return { .Encoding = Encoding::ZeroPoint, .Offset = range.Min + 32768 };
        }

        if (Fits<i16>(range)) return { .Encoding = Encoding::I16 };
        if (Fits<i32>(range)) return { .Encoding = Encoding::I32 };

        return { .Encoding = Encoding::I64 };
    }

    template<Format F>
    [[clang::always_inline]]
    constexpr auto Read(const Element<F> value)
    {
        if constexpr (F.Encoding == Encoding::F32) return value;
        else                                       return i64 { value } + F.Offset;
    }

    template<Format F, typename V>
    [[clang::always_inline]]
    constexpr Element<F> Write(const V value)
    {
        if (F.Encoding == Encoding::F32) return value;

        return static_cast<Element<F>>(value - F.Offset);
    }

    template<typename W>
    constexpr s00 BulletIndex(const s00 inputs, const s00 outputs, const s00 input, const s00 output)
    {
        return W::IsTransposed ? output * inputs + input : input * outputs + output;
    }

    template<typename D, typename S, typename Reader>
    LoadResult LoadAffine(Reader& reader, typename S::Parameters& parameters)
    {
        using W = D::WeightSpec;
        using B = D::BiasSpec;

        std::vector<typename W::Type> weights(D::InputSize * D::OutputSize);
        std::vector<typename B::Type> biases (               D::OutputSize);

        if (const LoadResult result = IO::ReadTensor<W>(reader, std::span(weights)); !result) return result;
        if (const LoadResult result = IO::ReadTensor<B>(reader, std::span(biases )); !result) return result;

        S::Pack(weights, biases, parameters);

        return {};
    }

    template<typename A, s00 I>
    consteval Interval ClampOf(const Interval bounds)
    {
        const Interval range = A::Signals[I].Range;

        return {
            .Min = bounds.Min > range.Min ? bounds.Min : std::numeric_limits<i16>::min(),
            .Max = bounds.Max < range.Max ? bounds.Max : std::numeric_limits<i16>::max()
        };
    }

}

#endif //MANTARAY_BACKEND_STAGE_FORMAT_H
