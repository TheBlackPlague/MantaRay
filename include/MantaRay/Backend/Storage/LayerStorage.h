//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BACKEND_LAYERSTORAGE_H
#define MANTARAY_BACKEND_LAYERSTORAGE_H

#include <array>
#include <tuple>
#include <utility>

#include "../../Common/Alignment.h"

namespace MantaRay::Backend
{

    template<typename Q, typename N>
    struct Storage;

    template<typename Q, s00 I, s00 O, typename A>
    struct Storage<Q, Accumulate<Layer<I, O, A>>>
    {

        ALIGN std::array<std::array<typename Q::FeatureType, O>, I> Weight;
        ALIGN std::array<           typename Q::FeatureType, O    >   Bias;

        template<typename F> void VisitParameters(F&& f)       { f(Weight); f(Bias); }
        template<typename F> void VisitParameters(F&& f) const { f(Weight); f(Bias); }

    };

    template<typename Q, s00 I, s00 O, typename A>
    struct Storage<Q, Layer<I, O, A>>
    {

        ALIGN std::array<std::array<typename Q::WeightType, I>, O> Weight;
        ALIGN std::array<           typename Q::   SumType,     O>   Bias;

        template<typename F> void VisitParameters(F&& f)       { f(Weight); f(Bias); }
        template<typename F> void VisitParameters(F&& f) const { f(Weight); f(Bias); }

    };

    template<typename Q, typename N>
    struct Storage<Q, Mirror<N>>
    {

        Storage<Q, N> Inner;

        template<typename F> void VisitParameters(F&& f)       { Inner.VisitParameters(f); }
        template<typename F> void VisitParameters(F&& f) const { Inner.VisitParameters(f); }

    };

    template<typename Q, typename N>
    struct Storage<Q, Residual<N>>
    {

        Storage<Q, N> Inner;

        template<typename F> void VisitParameters(F&& f)       { Inner.VisitParameters(f); }
        template<typename F> void VisitParameters(F&& f) const { Inner.VisitParameters(f); }

    };

    template<typename Q>
    struct Storage<Q, Concat>
    {

        template<typename F> void VisitParameters(F&&)       {}
        template<typename F> void VisitParameters(F&&) const {}

    };

    template<typename Q, typename... N>
    struct Storage<Q, Sequence<N...>>
    {

        std::tuple<Storage<Q, N>...> Nodes;

        template<typename F> void VisitParameters(F&& f)
        { std::apply([&](      auto&... node) { (node.VisitParameters(f), ...); }, Nodes); }
        template<typename F> void VisitParameters(F&& f) const
        { std::apply([&](const auto&... node) { (node.VisitParameters(f), ...); }, Nodes); }

    };

    template<typename Q, typename... N>
    struct Storage<Q, Parallel<N...>> : Storage<Q, Sequence<N...>> {};

}

#endif
