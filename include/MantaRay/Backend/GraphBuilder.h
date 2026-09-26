//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_GRAPHBUILDER_H
#define MANTARAY_BACKEND_GRAPHBUILDER_H

#include "Graph.h"
#include "Optimizer.h"
#include "Traits.h"

namespace MantaRay::Backend
{

    namespace Detail
    {

        template<typename T> struct NodeBudget;

        template<s00 I, s00 O, typename A, typename T>
        struct NodeBudget<Layer<I, O, A, T>> { constexpr static s00 Value = 3; };

        template<typename T>
        struct NodeBudget<Accumulate<T>> { constexpr static s00 Value = 3                       ; };
        template<typename T>
        struct NodeBudget<    Mirror<T>> { constexpr static s00 Value = 2 * NodeBudget<T>::Value; };
        template<typename T>
        struct NodeBudget<  Residual<T>> { constexpr static s00 Value = 1 + NodeBudget<T>::Value; };

        template<>
        struct NodeBudget<Concat> { constexpr static s00 Value = 1; };

        template<typename... T>
        struct NodeBudget<Sequence<T...>> { constexpr static s00 Value = (NodeBudget<T>::Value + ... + 0); };
        template<typename... T>
        struct NodeBudget<Parallel<T...>> : NodeBudget<Sequence<T...>> {};

        template<s00 Capacity>
        struct Edges
        {

            std::array<s00, Capacity> Values {};
            s00 Count = 0;

            constexpr static Edges Single(const s00 value)
            {
                Edges result;

                result.Values[0] = value;
                result.Count = 1;

                return result;
            }

            constexpr void Append(const Edges& other)
            {
                for (s00 i = 0; i < other.Count; i++) Values[Count++] = other.Values[i];
            }

        };

        template<typename A> struct ActivationDescriptor;

        template<>
        struct ActivationDescriptor<Identity>
        {

            template<typename N>
            constexpr static void Set(N& node) { node.Function = Node::Activation::Identity; }

        };

        template<i32 L, i32 H>
        struct ActivationDescriptor<ClippedReLU<L, H>>
        {

            template<typename N>
            constexpr static void Set(N& node)
            {
                node.Function = Node::Activation::Clipped;
                node.Minimum = L;
                node.Maximum = H;
            }

        };

        template<i32 L, i32 H>
        struct ActivationDescriptor<SquaredClippedReLU<L, H>>
        {

            template<typename N>
            constexpr static void Set(N& node)
            {
                node.Function = Node::Activation::Squared;
                node.Minimum = L;
                node.Maximum = H;
            }

        };

        template<typename A, typename G>
        constexpr s00 Activate(G& graph, const s00 input)
        {
            typename G::Instruction node;

            node.Code = Node::Operation::Activate;

            node.InputCount = 1;
            node.Inputs[0] = input;

            node.Size = graph.Nodes[input].Size;
            node.Unit = graph.Nodes[input].Unit;

            ActivationDescriptor<A>::Set(node);

            return graph.Append(node);
        }

        template<typename T> struct BuildNode;

        template<s00 I, s00 O, typename A, typename T>
        struct BuildNode<Layer<I, O, A, T>>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, const bool final)
            {
                typename Graph<Q, C>::Instruction affine;

                affine.Code = Node::Operation::Affine;

                affine.Size  =                     O;
                affine.Unit  = Node::Domain::Product;
                affine.Width =                     I;

                affine.InputCount = 1;
                affine.Inputs[0] = input.Values[0];

                affine.Parameter = graph.Parameters++;

                auto result = Activate<A>(graph, graph.Append(affine));

                if (!final) {
                    typename Graph<Q, C>::Instruction quantize;

                    quantize.Code = Node::Operation::Requantize;

                    quantize.Size = O;

                    quantize.InputCount = 1;
                    quantize.Inputs[0] = result;

                    result = graph.Append(quantize);
                }

                return Edges<C>::Single(result);
            }

        };

        template<s00 I, s00 O, typename A>
        struct BuildNode<Accumulate<Layer<I, O, A>>>
        {

            template<typename Q, s00 C>
            constexpr static s00 Perspective(
                Graph<Q, C>& graph, const Edges<C>& input, const s00 parameter, const s00 state
            )
            {
                typename Graph<Q, C>::Instruction affine;

                affine.Code = Node::Operation::Affine;

                affine.FeatureAffine = true;

                affine.Size  = O;
                affine.Width = I;

                affine.Parameter = parameter;

                affine.InputCount = 1;
                affine.Inputs[0] = input.Values[0];

                typename Graph<Q, C>::Instruction accumulator;

                accumulator.Code = Node::Operation::Accumulate;

                accumulator.Size = O;

                accumulator.State = state;

                accumulator.InputCount = 1;
                accumulator.Inputs[0] = graph.Append(affine);

                return Activate<A>(graph, graph.Append(accumulator));
            }

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, bool)
            {
                graph.Perspectives = 1;

                return Edges<C>::Single(Perspective(graph, input, graph.Parameters++, 0));
            }

        };

        template<typename T>
        struct BuildNode<Mirror<T>>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, bool)
            {
                graph.Perspectives = 2;

                const auto parameter = graph.Parameters++;

                Edges<C> result;

                result.Append(Edges<C>::Single(BuildNode<T>::Perspective(graph, input, parameter, 0)));
                result.Append(Edges<C>::Single(BuildNode<T>::Perspective(graph, input, parameter, 1)));

                return result;
            }

        };

        template<>
        struct BuildNode<Concat>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, bool)
            {
                typename Graph<Q, C>::Instruction node;

                node.Code = Node::Operation::Concat;

                node.Inputs     = input.Values;
                node.InputCount = input. Count;

                for (s00 i = 0; i < input.Count; i++) node.Size += graph.Nodes[input.Values[i]].Size;

                return Edges<C>::Single(graph.Append(node));
            }

        };

        template<typename... T>
        struct BuildNode<Sequence<T...>>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, Edges<C> input, const bool final)
            {
                s00 index = 0;
                ((input = BuildNode<T>::Run(graph, input, final && ++index == sizeof...(T))), ...);

                return input;
            }

        };

        template<typename... T>
        struct BuildNode<Parallel<T...>>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, bool)
            {
                Edges<C> result;
                (result.Append(BuildNode<T>::Run(graph, input, false)), ...);

                return result;
            }

        };

        template<typename T>
        struct BuildNode<Residual<T>>
        {

            template<typename Q, s00 C>
            constexpr static Edges<C> Run(Graph<Q, C>& graph, const Edges<C>& input, bool)
            {
                const auto branch = BuildNode<T>::Run(graph, input, false);

                typename Graph<Q, C>::Instruction node;

                node.Code = Node::Operation::Add;

                node.Size = graph.Nodes[input.Values[0]].Size;

                node.InputCount = 2;
                node.Inputs[0] =  input.Values[0];
                node.Inputs[1] = branch.Values[0];

                return Edges<C>::Single(graph.Append(node));
            }

        };

    }

    template<typename Architecture> struct GraphBuilder;

    template<typename Q, typename... T>
    struct GraphBuilder<Network<Q, T...>>
    {

        using Architecture = Network<Q, T...>;
        using Traits = ArchitectureTraits<Architecture>;

        static_assert(ValidArchitecture<Architecture>, "Invalid neural network composition");

        consteval static auto Build()
        {
            constexpr s00 capacity = Detail::NodeBudget<Sequence<T...>>::Value + 2;

            Graph<Q, capacity> graph;

            typename decltype(graph)::Instruction input;

            input.Size = Traits::Input::Size;

            auto result = Detail::BuildNode<Sequence<T...>>::Run(
                graph, Detail::Edges<capacity>::Single(graph.Append(input)), true
            );

            typename decltype(graph)::Instruction scale;

            scale.Code = Node::Operation::Scale;

            scale.Unit = Node::Domain::Product;
            scale.Size = graph.Nodes[result.Values[0]].Size;

            scale.InputCount = 1;
            scale.Inputs[0] = result.Values[0];

            graph.Output = graph.Append(scale);

            return graph;
        }

        constexpr static auto Unoptimized = Build();
        constexpr static auto   Optimized = Optimizer<Unoptimized>::Result;

    };

}

#endif
