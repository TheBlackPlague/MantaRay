//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_TRAITS_H
#define MANTARAY_BACKEND_TRAITS_H

#include <limits>
#include <tuple>
#include <type_traits>

#include "../Architecture/Layer.h"
#include "../Architecture/Network.h"
#include "../Architecture/Quantization.h"
#include "../Architecture/Activation/ClippedReLU.h"
#include "../Architecture/Activation/SquaredClippedReLU.h"
#include "../Architecture/Topology/Accumulate.h"
#include "../Architecture/Topology/Concat.h"
#include "../Architecture/Topology/Mirror.h"
#include "../Architecture/Topology/Parallel.h"
#include "../Architecture/Topology/Residual.h"
#include "../Architecture/Topology/Sequence.h"

namespace MantaRay::Backend
{

    struct InvalidShape {};

    template<s00 Width>
    struct VectorShape { constexpr static s00 Size = Width; };

    template<typename... Vectors>
    struct BundleShape { using Shapes = std::tuple<Vectors...>; };

    template<typename T>
    constexpr inline bool IsVector = false;

    template<s00 Width>
    constexpr inline bool IsVector<VectorShape<Width>> = true;

    template<typename A>
    struct ActivationTraits { constexpr static bool Valid = false; };

    template<>
    struct ActivationTraits<Identity> { constexpr static bool Valid = true; };

    template<i32 L, i32 H>
    struct ActivationTraits<ClippedReLU<L, H>> { constexpr static bool Valid = L < H; };

    template<i32 L, i32 H>
    struct ActivationTraits<SquaredClippedReLU<L, H>> { constexpr static bool Valid = L >= 0 && H > L; };

    template<typename A, typename Q>
    struct QuantizedActivation : std::false_type {};

    template<typename Q>
    struct QuantizedActivation<Identity, Q> : std::true_type {};

    template<i32 L, i32 H, typename Q>
    struct QuantizedActivation<ClippedReLU<L, H>, Q>
    {

        constexpr static bool value = [] {
            if (L >= H || Q::QA <= 0 || Q::QB <= 0) return false;

            constexpr i64 product = static_cast<i64>(Q::QA) * Q::QB;
            return static_cast<i64>(L) * Q::QA   >= std::numeric_limits<typename Q::FeatureType>::min() &&
                   static_cast<i64>(H) * Q::QA   <= std::numeric_limits<typename Q::FeatureType>::max() &&
                   static_cast<i64>(L) * product >= std::numeric_limits<typename Q::    SumType>::min() &&
                   static_cast<i64>(H) * product <= std::numeric_limits<typename Q::    SumType>::max()  ;
        }();

    };

    template<i32 L, i32 H, typename Q>
    struct QuantizedActivation<SquaredClippedReLU<L, H>, Q>
    {

        constexpr static bool value = [] {
            if constexpr (!QuantizedActivation<ClippedReLU<L, H>, Q>::value || L < 0) return false;
            else {
                constexpr i64 feature = std::numeric_limits<typename Q::FeatureType>::max();
                constexpr i64 sum     = std::numeric_limits<typename Q::    SumType>::max();

                return H <= feature / Q::QA / H && H <= sum / Q::QA / Q::QB / H;
            }
        }();

    };

    template<typename Q>
    struct QuantizationTraits { constexpr static bool Valid = false; };

    template<i32 A, i32 B, i32 Scale, typename F, typename W, typename S>
    struct QuantizationTraits<Quantization<A, B, Scale, F, W, S>>
    {

        constexpr static bool Valid =
            A > 0 && B > 0 && Scale > 0 &&
            A     <= std::numeric_limits<F>::max() &&
            B     <= std::numeric_limits<W>::max() &&
            Scale <= std::numeric_limits<S>::max() &&
            sizeof(S) >= sizeof(F) &&
            sizeof(S) >= sizeof(W) &&
            static_cast<i64>(A) * B <= std::numeric_limits<S>::max();

    };

    template<typename N>
    struct NodeTraits
    {

        using  Input = InvalidShape;
        using Output = InvalidShape;

        constexpr static bool Valid = false;

    };

    template<s00 I, s00 O, typename A, typename T>
    struct NodeTraits<Layer<I, O, A, T>>
    {

        using  Input = VectorShape<I>;
        using Output = VectorShape<O>;

        constexpr static bool Valid = I != 0 && O != 0 && ActivationTraits<A>::Valid && std::is_same_v<T, Affine>;

    };

    template<s00 I, s00 O, typename A>
    struct NodeTraits<Accumulate<Layer<I, O, A>>> : NodeTraits<Layer<I, O, A>> {};

    template<typename N>
    struct NodeTraits<Mirror<N>>
    {

        using  Input = NodeTraits<N>::Input;
        using Output = BundleShape<typename NodeTraits<N>::Output, typename NodeTraits<N>::Output>;

        constexpr static bool Valid = NodeTraits<N>::Valid;

    };

    template<typename N, typename Incoming>
    struct Connect
    {

        using Output = NodeTraits<N>::Output;

        constexpr static bool Valid = NodeTraits<N>::Valid && std::is_same_v<Incoming, typename NodeTraits<N>::Input>;

    };

    template<s00... Widths>
    struct Connect<Concat, BundleShape<VectorShape<Widths>...>>
    {

        using Output = VectorShape<(Widths + ... + 0)>;

        constexpr static bool Valid = sizeof...(Widths) != 0 && ((Widths != 0) && ...);

    };

    template<typename Incoming, typename... Nodes>
    struct Chain
    {

        using Output = Incoming;

        constexpr static bool Valid = true;

    };

    template<typename Incoming, typename Head, typename... Tail>
    struct Chain<Incoming, Head, Tail...>
    {

        using Step = Connect<Head, Incoming>;
        using Rest = Chain<typename Step::Output, Tail...>;
        using Output = Rest::Output;

        constexpr static bool Valid = Step::Valid && Rest::Valid;

    };

    template<typename Head, typename... Tail>
    struct NodeTraits<Sequence<Head, Tail...>>
    {

        using  Input = NodeTraits<Head>::Input;
        using Route  = Chain<Input, Head, Tail...>;
        using Output = Route::Output;

        constexpr static bool Valid = Route::Valid;

    };

    template<typename Head, typename... Tail>
    struct NodeTraits<Parallel<Head, Tail...>>
    {

        using  Input = NodeTraits<Head>::Input;
        using Output = BundleShape<typename NodeTraits<Head>::Output, typename NodeTraits<Tail>::Output...>;

        constexpr static bool Valid = NodeTraits<Head>::Valid && (Connect<Tail, Input>::Valid && ...);

    };

    template<typename N>
    struct NodeTraits<Residual<N>> : NodeTraits<N>
    {

        constexpr static bool Valid = NodeTraits<N>::Valid && IsVector<typename NodeTraits<N>::Input> &&
            std::is_same_v<typename NodeTraits<N>::Input, typename NodeTraits<N>::Output>;

    };

    template<typename N> constexpr inline bool StatelessNode                =             true;
    template<typename N> constexpr inline bool StatelessNode<Accumulate<N>> =            false;
    template<typename N> constexpr inline bool StatelessNode<    Mirror<N>> =            false;
    template<typename N> constexpr inline bool StatelessNode<  Residual<N>> = StatelessNode<N>;

    template<typename... N> constexpr inline bool StatelessNode<Sequence<N...>> = (StatelessNode<N> && ...);
    template<typename... N> constexpr inline bool StatelessNode<Parallel<N...>> = (StatelessNode<N> && ...);

    template<typename N>
    struct AccumulatorTraits
    {

        using AccumulatorLayer = void;

        constexpr static bool  HasAccumulator = false;
        constexpr static s00 PerspectiveCount =     0;

    };

    template<s00 I, s00 O, typename A>
    struct AccumulatorTraits<Accumulate<Layer<I, O, A>>>
    {

        using AccumulatorLayer = Layer<I, O, A>;

        constexpr static bool  HasAccumulator = true;
        constexpr static s00 PerspectiveCount =    1;

    };

    template<s00 I, s00 O, typename A>
    struct AccumulatorTraits<Mirror<Accumulate<Layer<I, O, A>>>> : AccumulatorTraits<Accumulate<Layer<I, O, A>>>
    {

        constexpr static s00 PerspectiveCount = 2;

    };

    template<typename N, typename Q>
    struct QuantizedNode : std::true_type {};

    template<s00 I, s00 O, typename A, typename T, typename Q>
    struct QuantizedNode<Layer<I, O, A, T>, Q> : QuantizedActivation<A, Q> {};

    template<typename N, typename Q> struct QuantizedNode<Accumulate<N>, Q> : QuantizedNode<N, Q> {};
    template<typename N, typename Q> struct QuantizedNode<    Mirror<N>, Q> : QuantizedNode<N, Q> {};
    template<typename N, typename Q> struct QuantizedNode<  Residual<N>, Q> : QuantizedNode<N, Q> {};

    template<typename... N, typename Q>
    struct QuantizedNode<Sequence<N...>, Q> : std::bool_constant<(QuantizedNode<N, Q>::value && ...)> {};

    template<typename... N, typename Q>
    struct QuantizedNode<Parallel<N...>, Q> : std::bool_constant<(QuantizedNode<N, Q>::value && ...)> {};

    template<typename A>
    struct ArchitectureTraits : NodeTraits<void> {};

    template<typename Q, typename First, typename... Rest>
    struct ArchitectureTraits<Network<Q, First, Rest...>> : AccumulatorTraits<First>
    {

        using Quantization = Q;

        using Head = First;
        using Tail = std::tuple<Rest...>;

        using Shape = NodeTraits<Sequence<First, Rest...>>;

        using  Input = Shape:: Input;
        using Output = Shape::Output;

        constexpr static bool Valid = [] {
            if constexpr (!QuantizationTraits<Q>::Valid) return false;
            else return Shape::Valid && IsVector<Output> &&
                (StatelessNode<First> || AccumulatorTraits<First>::HasAccumulator) &&
                (StatelessNode<Rest> && ...) && QuantizedNode<Sequence<First, Rest...>, Q>::value;
        }();

    };

    template<typename A>
    concept ValidArchitecture = ArchitectureTraits<A>::Valid;

}

#endif
