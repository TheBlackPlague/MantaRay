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

    template<s00 N>
    struct VectorShape { constexpr static s00 Size = N; };

    template<typename... S>
    struct BundleShape { using Shapes = std::tuple<S...>; };

    template<typename _>
    struct ActivationTraits { constexpr static bool Valid = false; };

    template<>
    struct ActivationTraits<Identity> { constexpr static bool Valid = true; };

    template<i32 L, i32 H>
    struct ActivationTraits<ClippedReLU<L, H>> { constexpr static bool Valid = L < H; };

    template<i32 L, i32 H>
    struct ActivationTraits<SquaredClippedReLU<L, H>> { constexpr static bool Valid = 0 <= L && L < H; };

    template<typename _>
    struct TransformTraits { constexpr static bool Valid = false; };

    template<>
    struct TransformTraits<Affine> { constexpr static bool Valid = true; };

    template<typename _>
    struct NodeTraits
    {

        constexpr static bool Valid = false;

        using  Input = InvalidShape;
        using Output = InvalidShape;

    };

    template<s00 I, s00 O, typename A, typename T>
    struct NodeTraits<Layer<I, O, A, T>>
    {

        constexpr static bool Valid = I > 0 && O > 0 && ActivationTraits<A>::Valid && TransformTraits<T>::Valid;

        using  Input = VectorShape<I>;
        using Output = VectorShape<O>;

    };

    template<typename N>
    struct NodeTraits<Accumulate<N>> : NodeTraits<N> { constexpr static bool Valid = false; };

    template<s00 I, s00 O, typename A>
    struct NodeTraits<Accumulate<Layer<I, O, A>>> : NodeTraits<Layer<I, O, A>> {};

    template<typename N>
    struct NodeTraits<Mirror<N>>
    {

        constexpr static bool Valid = NodeTraits<N>::Valid;

        using  Input = NodeTraits<N>::Input;
        using Output = BundleShape<
            typename NodeTraits<N>::Output,
            typename NodeTraits<N>::Output
        >;

    };

    template<typename N, typename Incoming>
    struct Connect
    {

        constexpr static bool Valid = NodeTraits<N>::Valid && std::is_same_v<Incoming, typename NodeTraits<N>::Input>;

        using Output = NodeTraits<N>::Output;

    };

    template<s00... Ns>
    struct Connect<Concat, BundleShape<VectorShape<Ns>...>>
    {

        constexpr static bool Valid = sizeof...(Ns) > 0 && ((Ns > 0) && ...);

        using Output = VectorShape<(Ns + ... + 0)>;

    };

    template<typename Incoming, typename... Ns>
    struct Chain;

    template<typename Incoming>
    struct Chain<Incoming>
    {

        constexpr static bool Valid = true;

        using Output = Incoming;

    };

    template<typename Incoming, typename N, typename... Ns>
    struct Chain<Incoming, N, Ns...>
    {

        using Connection = Connect<N, Incoming>;
        using Rest = Chain<typename Connection::Output, Ns...>;

        constexpr static bool Valid = Connection::Valid && Rest::Valid;

        using Output = Rest::Output;

    };

    template<>
    struct NodeTraits<Sequence<>>
    {

        constexpr static bool Valid = false;

        using  Input = InvalidShape;
        using Output = InvalidShape;

    };

    template<typename N, typename... Ns>
    struct NodeTraits<Sequence<N, Ns...>>
    {
        using First = NodeTraits<N>;
        using Rest  = Chain<typename First::Output, Ns...>;

        constexpr static bool Valid = First::Valid && Rest::Valid;

        using Input  = First::Input;
        using Output = Rest::Output;
    };

    template<>
    struct NodeTraits<Parallel<>> : NodeTraits<Sequence<>> {};

    template<typename N, typename... Ns>
    struct NodeTraits<Parallel<N, Ns...>>
    {

        using  Input = NodeTraits<N>::Input;
        using Output = BundleShape<typename NodeTraits<N>::Output, typename NodeTraits<Ns>::Output...>;

        constexpr static bool Valid = NodeTraits<N>::Valid && (NodeTraits<Ns>::Valid && ...) &&
            (std::is_same_v<Input, typename NodeTraits<Ns>::Input> && ...);

    };

    template<typename S>
    constexpr inline bool IsVector = false;

    template<s00 N>
    constexpr inline bool IsVector<VectorShape<N>> = true;

    template<typename N>
    struct NodeTraits<Residual<N>> : NodeTraits<N>
    {

        constexpr static bool Valid = NodeTraits<N>::Valid && IsVector<typename NodeTraits<N>::Input> &&
            std::is_same_v<typename NodeTraits<N>::Input, typename NodeTraits<N>::Output>;

    };

    template<typename    N > constexpr inline bool StatelessNode                    =  true                     ;
    template<typename    N > constexpr inline bool StatelessNode<Accumulate<N    >> = false                     ;
    template<typename    N > constexpr inline bool StatelessNode<Mirror    <N    >> = false                     ;
    template<typename... Ns> constexpr inline bool StatelessNode<Sequence  <Ns...>> = (StatelessNode<Ns> && ...);
    template<typename... Ns> constexpr inline bool StatelessNode<Parallel  <Ns...>> = (StatelessNode<Ns> && ...);
    template<typename    N > constexpr inline bool StatelessNode<Residual  <N    >> =  StatelessNode<N >        ;

    template<typename _>
    struct AccumulatorTraits
    {

        constexpr static bool   HasAccumulator = false;
        constexpr static s00 PerspectiveCount =     0;

        using AccumulatorLayer = void;

    };

    template<s00 I, s00 O, typename A>
    struct AccumulatorTraits<Accumulate<Layer<I, O, A>>>
    {

        constexpr static bool   HasAccumulator = true;
        constexpr static s00 PerspectiveCount =    1;

        using AccumulatorLayer = Layer<I, O, A>;

    };

    template<s00 I, s00 O, typename A>
    struct AccumulatorTraits<Mirror<Accumulate<Layer<I, O, A>>>>
    {

        constexpr static bool   HasAccumulator = true;
        constexpr static s00 PerspectiveCount =    2;

        using AccumulatorLayer = Layer<I, O, A>;

    };

    template<typename _> struct QuantizationTraits { constexpr static bool Valid = false; };

    template<i32 A, i32 B, i32 S, QuantizedInteger F, QuantizedInteger W, QuantizedInteger U>
    struct QuantizationTraits<Quantization<A, B, S, F, W, U>>
    {

        template<QuantizedInteger Q>
        using NumericLimits = std::numeric_limits<Q>;

        constexpr static bool Valid = A > 0 && B > 0 && S > 0 &&
            A <= NumericLimits<F>::max() && B <= NumericLimits<W>::max() && S <= NumericLimits<U>::max() &&
            sizeof(U) >= sizeof(F) && sizeof(U) >= sizeof(W) &&
            static_cast<i64>(A) * B <= NumericLimits<U>::max();

    };

    template<typename A, typename _> struct QuantizedActivation : std::bool_constant<ActivationTraits<A>::Valid> {};

    template<i32 L, i32 H, typename Q> struct QuantizedActivation<ClippedReLU<L, H>, Q> : std::bool_constant<
        ActivationTraits<ClippedReLU<L, H>>::Valid &&

        static_cast<i64>(L) * Q::QA         >= std::numeric_limits<typename Q::FeatureType>::min() &&
        static_cast<i64>(H) * Q::QA         <= std::numeric_limits<typename Q::FeatureType>::max() &&
        static_cast<i64>(L) * Q::QA * Q::QB >= std::numeric_limits<typename Q::    SumType>::min() &&
        static_cast<i64>(H) * Q::QA * Q::QB <= std::numeric_limits<typename Q::    SumType>::max()
    > {};

    template<i32 L, i32 H, typename Q> struct QuantizedActivation<SquaredClippedReLU<L, H>, Q> : std::bool_constant<
        ActivationTraits<SquaredClippedReLU<L, H>   >::Valid &&
        QuantizedActivation<    ClippedReLU<L, H>, Q>::value &&

        H <= std::numeric_limits<typename Q::FeatureType>::max() / Q::QA /         H &&
        H <= std::numeric_limits<typename Q::    SumType>::max() / Q::QA / Q::QB / H &&
        static_cast<i64>(H) * Q::QA * Q::QB <= 3037000499LL
    > {};

    template<typename N, typename Q> struct QuantizedNode : std::true_type {};

    template<s00 I, s00 O, typename A, typename T, typename Q>
    struct QuantizedNode<Layer<I, O, A, T>, Q> : QuantizedActivation<A, Q> {};

    template<typename    N , typename Q> struct QuantizedNode<Accumulate<N>, Q> : QuantizedNode<N, Q> {};
    template<typename    N , typename Q> struct QuantizedNode<Mirror    <N>, Q> : QuantizedNode<N, Q> {};
    template<typename    N , typename Q> struct QuantizedNode<Residual  <N>, Q> : QuantizedNode<N, Q> {};

    template<typename... Ns, typename Q> struct QuantizedNode<Parallel<Ns...>, Q> :
        std::bool_constant<(QuantizedNode<Ns, Q>::value && ...)> {};

    template<typename... Ns, typename Q> struct QuantizedNode<Sequence<Ns...>, Q> :
        std::bool_constant<(QuantizedNode<Ns, Q>::value && ...)> {};

    template<typename _>
    struct ArchitectureTraits
    {

        constexpr static bool Valid = false;

        using  Input = InvalidShape;
        using Output = InvalidShape;

    };

    template<typename Q> struct ArchitectureTraits<Network<Q>> : ArchitectureTraits<void> {};

    template<typename Q, typename N, typename... Ns>
    struct ArchitectureTraits<Network<Q, N, Ns...>> : AccumulatorTraits<N>
    {

        using Quantization = Q;

        using Head =            N     ;
        using Tail = std::tuple<Ns...>;

        using  Input = NodeTraits<Sequence<N, Ns...>>:: Input;
        using Output = NodeTraits<Sequence<N, Ns...>>::Output;

        constexpr static bool Valid = [] {
            if constexpr (!QuantizationTraits<Q>::Valid) return false;
            else return NodeTraits<Sequence<N, Ns...>>::Valid && IsVector<Output> &&
                        (StatelessNode<N> || AccumulatorTraits<N>::HasAccumulator) &&
                        (StatelessNode<Ns> && ...) && QuantizedNode<N, Q>::value &&
                        (QuantizedNode<Ns, Q>::value && ...);
        }();

    };

    template<typename A> concept ValidArchitecture = ArchitectureTraits<A>::Valid;

}

#endif
