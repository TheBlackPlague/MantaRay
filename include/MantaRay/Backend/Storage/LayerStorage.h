//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_STORAGE_LAYERSTORAGE_H
#define MANTARAY_BACKEND_STORAGE_LAYERSTORAGE_H

#include <array>
#include <tuple>
#include <utility>

#include "../Traits.h"
#include "../../Common/Alignment.h"
#include "../../Common/Container.h"

namespace MantaRay::Backend
{

    template<typename Q, typename N>
    struct Storage;

    template<typename Q, s00 Input, s00 Output, typename A, typename T>
    struct Storage<Q, Layer<Input, Output, A, T>>
    {

        constexpr static s00 ParameterCount = 1;

        ALIGN std::array<Array<typename Q::WeightType, Input>, Output> Weight;
        ALIGN Array<           typename Q::   SumType,         Output>   Bias;

        template<s00 Index> auto& Get()       { static_assert(Index == 0); return *this; }
        template<s00 Index> auto& Get() const { static_assert(Index == 0); return *this; }

        template<typename F> void VisitParameters(F&& visitor)       { visitor(Weight); visitor(Bias); }
        template<typename F> void VisitParameters(F&& visitor) const { visitor(Weight); visitor(Bias); }

    };

    template<typename Q, s00 Input, s00 Output, typename A, typename T>
    struct Storage<Q, Accumulate<Layer<Input, Output, A, T>>>
    {

        constexpr static s00 ParameterCount = 1;

        ALIGN std::array<Array<typename Q::FeatureType, Output>, Input> Weight;
        ALIGN Array<           typename Q::FeatureType, Output        >   Bias;

        template<s00 Index> auto& Get()       { static_assert(Index == 0); return *this; }
        template<s00 Index> auto& Get() const { static_assert(Index == 0); return *this; }

        template<typename F> void VisitParameters(F&& visitor)       { visitor(Weight); visitor(Bias); }
        template<typename F> void VisitParameters(F&& visitor) const { visitor(Weight); visitor(Bias); }

    };

    template<typename Q, typename N>
    struct Storage<Q, Mirror<N>>
    {

        constexpr static s00 ParameterCount = Storage<Q, N>::ParameterCount;

        Storage<Q, N> Inner;

        template<s00 Index> auto& Get()       { return Inner.template Get<Index>(); }
        template<s00 Index> auto& Get() const { return Inner.template Get<Index>(); }

        template<typename F> void VisitParameters(F&& visitor)       { Inner.VisitParameters(visitor); }
        template<typename F> void VisitParameters(F&& visitor) const { Inner.VisitParameters(visitor); }

    };

    template<typename Q, typename N>
    struct Storage<Q, Residual<N>> : Storage<Q, Mirror<N>> {};

    template<typename Q>
    struct Storage<Q, Concat>
    {

        constexpr static s00 ParameterCount = 0;

        template<typename F> void VisitParameters(F&&)       {}
        template<typename F> void VisitParameters(F&&) const {}

    };

    template<typename Q>
    struct Storage<Q, Sequence<>>
    {

        constexpr static s00 ParameterCount = 0;

        auto Nodes()       { return std::tuple<> {}; }
        auto Nodes() const { return std::tuple<> {}; }

        template<typename F> void VisitParameters(F&&)       {}
        template<typename F> void VisitParameters(F&&) const {}

    };

    template<typename Q, typename Head, typename... Tail>
    struct Storage<Q, Sequence<Head, Tail...>>
    {

        using HeadStorage = Storage<Q, Head>             ;
        using TailStorage = Storage<Q, Sequence<Tail...>>;

        constexpr static s00 ParameterCount = HeadStorage::ParameterCount + TailStorage::ParameterCount;

        NO_UNIQUE_ADDRESS
        HeadStorage First;

        NO_UNIQUE_ADDRESS
        TailStorage Rest;

        auto Nodes()       { return std::tuple_cat(std::tie(First), Rest.Nodes()); }
        auto Nodes() const { return std::tuple_cat(std::tie(First), Rest.Nodes()); }

        template<s00 Index>
        auto& Get()
        {
            static_assert(Index < ParameterCount);

            if constexpr (Index < HeadStorage::ParameterCount)
                 return First.template Get<Index                              >();
            else return  Rest.template Get<Index - HeadStorage::ParameterCount>();
        }

        template<s00 Index>
        auto& Get() const
        {
            static_assert(Index < ParameterCount);

            if constexpr (Index < HeadStorage::ParameterCount)
                 return First.template Get<Index                              >();
            else return Rest .template Get<Index - HeadStorage::ParameterCount>();
        }

        template<typename F>
        void VisitParameters(F&& visitor)
        {
            First.VisitParameters(visitor);
            Rest .VisitParameters(visitor);
        }

        template<typename F>
        void VisitParameters(F&& visitor) const
        {
            First.VisitParameters(visitor);
            Rest .VisitParameters(visitor);
        }

    };

    template<typename Q, typename... Nodes>
    struct Storage<Q, Parallel<Nodes...>> : Storage<Q, Sequence<Nodes...>> {};

}

#endif
