// Copyright (c) 2026 MantaRay authors. Licensed under MIT.
#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "Kernel/Activation.h"
#include "Kernel/Requantize.h"
#include "Kernel/Fused/ActivateConcatAffine.h"
#include "Storage/NetworkStorage.h"

namespace MantaRay::Backend
{

    template<class A, class Q, s00 N>
    struct ActivatedView
    {

        static constexpr s00           Size =     N;
        static constexpr bool ProductDomain = false;

        const Array<typename Q::FeatureType, N>& Values;

        auto operator[](s00 i) const { return Activation<A, Q>::Apply(Values[i]); }

    };

    template<class Q, s00 N, bool Product = false>
    struct Value
    {

        static constexpr s00           Size =       N;
        static constexpr bool ProductDomain = Product;

        using Element = std::conditional_t<Product, typename Q::SumType, typename Q::FeatureType>;

        alignas(64) Array<Element, N> Values {};

        auto operator[](s00 i) const { return Values[i]; }

    };

    template<class... Views>
    struct ConcatView
    {

        static constexpr s00           Size = (Views::Size + ...);
        static constexpr bool ProductDomain =               false;

        std::tuple<Views...> ViewsTuple;

        auto operator[](s00 i) const
        {
            using Element = decltype(std::get<0>(ViewsTuple)[0]);

            Element result {};

            std::apply([&]<typename... T0>(const T0&... view) {
                (
                    (
                        i < std::remove_cvref_t<T0>::Size ?
                        (void)(result = view[i], i = Size) :
                        (void)(i -= std::remove_cvref_t<T0>::Size)
                    ),
                    ...
                );
            }, ViewsTuple);

            return result;
        }

    };

    template<class Transform, class Q, class L>
    struct TransformKernel;

    template<class Q, s00 I, s00 O, class A>
    struct TransformKernel<Affine, Q, Layer<I, O, A>>
    {

        using L = Layer<I, O, A>;

        template<class Input>
        static auto Run(const Storage<Q, L>& storage, const Input& input)
        {
            Array<typename Q::SumType, O> result {};

            for (s00 out = 0; out < O; ++out) {
                auto sum = storage.Bias[out];

                for (s00 in = 0; in < I; ++in) {
                    sum = WrapAdd(
                        sum,
                        WrapMul(
                            static_cast<Q::SumType>(input              [in]),
                            static_cast<Q::SumType>(storage.Weight[out][in])
                        )
                    );
                }

                result[out] = sum;
            }

            return result;
        }

        template<class Previous, s00 N>
        static auto Run(
            const Storage<Q, L>& storage,
            const ConcatView<ActivatedView<Previous, Q, N>, ActivatedView<Previous, Q, N>>& input
        )
        {
            static_assert(I == 2 * N);

            return Kernel::ActivateConcatAffine<Previous, Q, N, O>(
                std::get<0>(input.ViewsTuple).Values,
                std::get<1>(input.ViewsTuple).Values,
                storage.Weight,
                storage.Bias
            );
        }

    };

    template<class Node, class Q>
    struct Lower;

    template<class Q, class... Stages>
    struct LowerSequence
    {

        template<bool Final, s00 Index = 0, class Storages, class Input>
        static auto Run(const Storages& storage, const Input& input)
        {
            if constexpr (Index == sizeof...(Stages)) return input;

            using Node = std::tuple_element_t<Index, std::tuple<Stages...>>;

            auto output = Lower<Node, Q>::template Run<Final && Index + 1 ==
                          sizeof...(Stages)>(std::get<Index>(storage), input);

            return Run<Final, Index + 1>(storage, output);
        }

    };

    template<s00 I, s00 O, class A, class T, class Q>
    struct Lower<Layer<I, O, A, T>, Q>
    {

        template<bool Final, class Input>
        static auto Run(const Storage<Q, Layer<I, O, A, T>>& storage, const Input& input)
        {
            static_assert(Input::Size == I);

            const auto sums = TransformKernel<T, Q, Layer<I, O, A, T>>::Run(storage, input);

            Value<Q, O, Final> output;
            for (s00 i = 0; i < O; ++i) {
                const auto activated = Activation<A, Q>::ApplySum(sums[i]);

                if constexpr (Final) output.Values[i] =                       activated ;
                else                 output.Values[i] = Kernel::Requantize<Q>(activated);
            }

            return output;
        }

    };

    template<class Q>
    struct Lower<Concat, Q>
    {

        template<bool, class... Views>
        static auto Run(const Storage<Q, Concat>&, const std::tuple<Views...>& input)
        { return ConcatView<Views...>{input}; }

    };

    template<class... Stages, class Q>
    struct Lower<Sequence<Stages...>, Q>
    {

        template<bool Final, class Input>
        static auto Run(const Storage<Q, Sequence<Stages...>>& storage, const Input& input)
        { return LowerSequence<Q, Stages...>::template Run<Final>(storage.Nodes, input); }

    };

    template<class... Branches, class Q>
    struct Lower<Parallel<Branches...>, Q>
    {

        template<bool, class Input>
        static auto Run(const Storage<Q, Parallel<Branches...>>& storage, const Input& input)
        {
            return [&]<s00... Is>(std::index_sequence<Is...>) {
                return std::tuple{Lower<Branches, Q>::template Run<false>(std::get<Is>(storage.Nodes), input)...};
            }(std::index_sequence_for<Branches...> {});
        }

    };

    template<class Branch, class Q>
    struct Lower<Residual<Branch>, Q>
    {

        template<bool, class Input>
        static auto Run(const Storage<Q, Residual<Branch>>& storage, const Input& input)
        {
            const auto branch = Lower<Branch, Q>::template Run<false>(storage.Inner, input);

            Value<Q, Input::Size> output;
            for (s00 i = 0; i < Input::Size; ++i) output.Values[i] = WrapAdd(
                static_cast<Q::FeatureType>(input[i]),
                branch[i]
            );

            return output;
        }

    };

    template<class Q, class Result>
    auto ScaleResult(const Result& result)
    {
        Array<typename Q::SumType, Result::Size> output;

        for (s00 i = 0; i < Result::Size; ++i) {
            if constexpr (Result::ProductDomain)
                 output[i] = Kernel::FinalScale<Q>(result[i]);
            else output[i] = WrapMul(
                static_cast<Q::SumType>(result[i]),
                static_cast<Q::SumType>(Q::OutputScale)
            ) / Q::QA;
        }

        if constexpr (Result::Size == 1) return output[0];

        return output;
    }

}
