//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#pragma once

#include <limits>
#include <tuple>
#include <type_traits>

#include "../Architecture/Network.h"

namespace MantaRay::Backend
{

    struct InvalidShape {};

    template<std::size_t N>
    struct VectorShape { static constexpr std::size_t Size = N; };

    template<typename... S>
    struct BundleShape { using Shapes = std::tuple<S...>; };

    template<typename _>
    struct ActivationTraits { static constexpr bool Valid = false; };

    template<>
    struct ActivationTraits<Identity> { static constexpr bool Valid = true; };

    template<int L, int H>
    struct ActivationTraits<ClippedReLU<L, H>> { static constexpr bool Valid = L < H; };

    template<int L, int H>
    struct ActivationTraits<SquaredClippedReLU<L, H>> { static constexpr bool Valid = 0 <= L && L < H; };

    template<typename _>
    struct TransformTraits { static constexpr bool Valid = false; };

    template<>
    struct TransformTraits<Affine> { static constexpr bool Valid = true; };

    template<typename _>
    struct NodeTraits
    {

        static constexpr bool Valid = false;

        using  Input = InvalidShape;
        using Output = InvalidShape;

    };

    template<std::size_t I, std::size_t O, typename A, typename T>
    struct NodeTraits<Layer<I, O, A, T>>
    {

        static constexpr bool Valid = I > 0 && O > 0 && ActivationTraits<A>::Valid && TransformTraits<T>::Valid;

        using  Input = VectorShape<I>;
        using Output = VectorShape<O>;

    };

    template<typename N>
    struct NodeTraits<Accumulate<N>> : NodeTraits<N> { static constexpr bool Valid = false; };

    template<std::size_t I, std::size_t O, typename A>
    struct NodeTraits<Accumulate<Layer<I, O, A>>> : NodeTraits<Layer<I, O, A>> {};

    template<typename N>
    struct NodeTraits<Mirror<N>>
    {

        static constexpr bool Valid = NodeTraits<N>::Valid;

        using  Input = NodeTraits<N>::Input;
        using Output = BundleShape<
            typename NodeTraits<N>::Output,
            typename NodeTraits<N>::Output
        >;

    };

    template<typename N, typename Incoming>
    struct Connect
    {

        static constexpr bool Valid = NodeTraits<N>::Valid && std::is_same_v<Incoming, typename NodeTraits<N>::Input>;

        using Output = NodeTraits<N>::Output;

    };

    template<std::size_t... Ns>
    struct Connect<Concat, BundleShape<VectorShape<Ns>...>>
    {

        static constexpr bool Valid = sizeof...(Ns) > 0 && ((Ns > 0) && ...);

        using Output = VectorShape<(Ns + ... + 0)>;

    };

    template<typename Incoming, typename... Ns>
    struct Chain;

    template<typename Incoming>
    struct Chain<Incoming>
    {

        static constexpr bool Valid = true;

        using Output = Incoming;

    };

    template<typename Incoming, typename N, typename... Ns>

    struct Chain<Incoming, N, Ns...>
    {

        using Connection = Connect<N, Incoming>;
        using Rest = Chain<typename Connection::Output, Ns...>;

        static constexpr bool Valid = Connection::Valid && Rest::Valid;

        using Output = Rest::Output;

    };

    template<>
    struct NodeTraits<Sequence<>>
    {

        static constexpr bool Valid = false;

        using  Input = InvalidShape;
        using Output = InvalidShape;

    };

    template<typename N, typename... Ns>
    struct NodeTraits<Sequence<N, Ns...>>
    {
        using First = NodeTraits<N>;
        using Rest  = Chain<typename First::Output, Ns...>;

        static constexpr bool Valid = First::Valid && Rest::Valid;

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

        static constexpr bool Valid = NodeTraits<N>::Valid && (NodeTraits<Ns>::Valid && ...) &&
            (std::is_same_v<Input, typename NodeTraits<Ns>::Input> && ...);

    };

    template<typename S>
    inline constexpr bool IsVector = false;

    template<std::size_t N>
    inline constexpr bool IsVector<VectorShape<N>> = true;

    template<typename N>
    struct NodeTraits<Residual<N>> : NodeTraits<N>
    {

        static constexpr bool Valid = NodeTraits<N>::Valid && IsVector<typename NodeTraits<N>::Input> &&
            std::is_same_v<typename NodeTraits<N>::Input, typename NodeTraits<N>::Output>;

    };

    template<typename    N > inline constexpr bool StatelessNode                    =  true                     ;
    template<typename    N > inline constexpr bool StatelessNode<Accumulate<N    >> = false                     ;
    template<typename    N > inline constexpr bool StatelessNode<Mirror    <N    >> = false                     ;
    template<typename... Ns> inline constexpr bool StatelessNode<Sequence  <Ns...>> = (StatelessNode<Ns> && ...);
    template<typename... Ns> inline constexpr bool StatelessNode<Parallel  <Ns...>> = (StatelessNode<Ns> && ...);
    template<typename    N > inline constexpr bool StatelessNode<Residual  <N    >> =  StatelessNode<N >        ;

    template<typename _>
    struct AccumulatorTraits
    {

        static constexpr bool          HasAccumulator = false;
        static constexpr std::size_t PerspectiveCount =     0;

        using AccumulatorLayer = void;

    };

    template<std::size_t I, std::size_t O, typename A>
    struct AccumulatorTraits<Accumulate<Layer<I, O, A>>>
    {

        static constexpr bool          HasAccumulator = true;
        static constexpr std::size_t PerspectiveCount =    1;

        using AccumulatorLayer = Layer<I, O, A>;

    };

    template<std::size_t I, std::size_t O, typename A>
    struct AccumulatorTraits<Mirror<Accumulate<Layer<I, O, A>>>>
    {

        static constexpr bool          HasAccumulator = true;
        static constexpr std::size_t PerspectiveCount =    2;

        using AccumulatorLayer = Layer<I, O, A>;

    };

    template<typename _> struct QuantizationTraits { static constexpr bool Valid = false; };

    template<int A, int B, int S, QuantizedInteger F, QuantizedInteger W, QuantizedInteger U>
    struct QuantizationTraits<Quantization<A, B, S, F, W, U>>
    {

        template<QuantizedInteger Q>
        using NumericLimits = std::numeric_limits<Q>;

        static constexpr bool Valid = A > 0 && B > 0 && S > 0 &&
            A <= NumericLimits<F>::max() && B <= NumericLimits<W>::max() && S <= NumericLimits<U>::max() &&
            sizeof(U) >= sizeof(F) && sizeof(U) >= sizeof(W) &&
            static_cast<std::int64_t>(A) * B <= NumericLimits<U>::max();

    };

    template<typename A, typename _> struct QuantizedActivation : std::bool_constant<ActivationTraits<A>::Valid> {};

    template<int L, int H, typename Q> struct QuantizedActivation<ClippedReLU<L, H>, Q> : std::bool_constant<
        ActivationTraits<ClippedReLU<L, H>>::Valid &&

        static_cast<std::int64_t>(L) * Q::QA         >= std::numeric_limits<typename Q::FeatureType>::min() &&
        static_cast<std::int64_t>(H) * Q::QA         <= std::numeric_limits<typename Q::FeatureType>::max() &&
        static_cast<std::int64_t>(L) * Q::QA * Q::QB >= std::numeric_limits<typename Q::    SumType>::min() &&
        static_cast<std::int64_t>(H) * Q::QA * Q::QB <= std::numeric_limits<typename Q::    SumType>::max()
    > {};

    template<int L, int H, typename Q> struct QuantizedActivation<SquaredClippedReLU<L, H>, Q> : std::bool_constant<
        ActivationTraits<SquaredClippedReLU<L, H>   >::Valid &&
        QuantizedActivation<    ClippedReLU<L, H>, Q>::value &&

        H <= std::numeric_limits<typename Q::FeatureType>::max() / Q::QA /         H &&
        H <= std::numeric_limits<typename Q::    SumType>::max() / Q::QA / Q::QB / H &&
        static_cast<std::int64_t>(H) * Q::QA * Q::QB <= 3037000499LL
    > {};

    template<typename _, typename __> struct QuantizedNode : std::true_type {};

    template<std::size_t I, std::size_t O, typename A, typename T, typename Q>
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

        static constexpr bool Valid = false;

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

        static constexpr bool Valid = [] {
            if constexpr (!QuantizationTraits<Q>::Valid) return false;

            return NodeTraits<Sequence<N, Ns...>>::Valid && IsVector<Output>   &&
                   (StatelessNode<N > || AccumulatorTraits<N>::HasAccumulator) &&
                   (StatelessNode<Ns> && ...                                 ) &&
                    QuantizedNode<N , Q>::value                                &&
                   (QuantizedNode<Ns, Q>::value && ...                       )  ;
        }();

    };

    template<typename A> concept ValidArchitecture = ArchitectureTraits<A>::Valid;

}
