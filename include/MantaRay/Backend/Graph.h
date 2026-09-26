//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_GRAPH_H
#define MANTARAY_BACKEND_GRAPH_H

#include <limits>
#include <utility>

#include "Node/Operation.h"
#include "Traits.h"

namespace MantaRay::Backend
{

    template<typename Q, s00 Capacity>
    struct Graph
    {

        using Quantization = Q;
        using Instruction  = Node::Instruction<Capacity>;

        constexpr static s00 Limit = Capacity;

        std::array<Instruction, Capacity> Nodes {};

        s00        Count = 0;
        s00       Output = 0;
        s00   Parameters = 0;
        s00 Perspectives = 0;

        constexpr s00 Append(const Instruction& instruction)
        {
            if (Count == Capacity) __builtin_unreachable();

            Nodes[Count] = instruction;

            return Count++;
        }

        constexpr bool operator ==(const Graph&) const = default;

        constexpr bool Valid() const
        {
            using enum Node::Operation;
            using enum Node::Domain   ;

            if (Count == 0 || Count > Capacity || Output >= Count) return false;

            if (!QuantizationTraits<Q>::Valid || Perspectives > 2) return false;

            const auto activationValid = [](const Instruction& node, const bool feature) {
                using F = Q::FeatureType;
                using S = Q::    SumType;

                if (node.Function == Node::Activation::Identity) return true;

                if (node.Function != Node::Activation::Clipped &&
                    node.Function != Node::Activation::Squared) return false;

                if (node.Minimum >= node.Maximum) return false;

                const i64 scale = feature ? Q::QA : i64 { Q::QA } * Q::QB;

                const i64 lower = i64 { node.Minimum } * scale;
                const i64 upper = i64 { node.Maximum } * scale;

                const i64 minimum = feature ? std::numeric_limits<F>::min() : std::numeric_limits<S>::min();
                const i64 maximum = feature ? std::numeric_limits<F>::max() : std::numeric_limits<S>::max();

                if (lower < minimum || upper > maximum) return false;

                return node.Function != Node::Activation::Squared ?
                       true : lower >= 0 && upper <= 3037000499LL && node.Maximum <= maximum / scale / node.Maximum;
            };

            s00 inputSize = 0;
            s00 stateSize = 0;

            for (s00 i = 0; i < Count; i++) {
                const auto& node = Nodes[i];

                if (node.Size == 0 || node.InputCount > Capacity) return false;

                if (node.Unit != Feature && node.Unit != Product) return false;

                if (node.Code < Input || node.Code > ActivateConcatAffine) return false;

                if (node.FeatureAffine && node.Code != Affine) return false;


                if (node.NarrowSquare && ((node.Code != Activate && node.Code != ActivateConcatAffine) ||
                    node.Function != Node::Activation::Squared ||
                    (node.Unit != Feature && node.Code != ActivateConcatAffine) ||
                    !std::is_same_v<typename Q::FeatureType, i16> || node.Minimum < 0 ||
                    i64 { node.Maximum } * Q::QA > 181))
                    return false;

                for (s00 j = 0; j < node.InputCount; j++) if (node.Inputs[j] >= i) return false;

                const auto unary = [&] { return node.InputCount == 1; };

                const auto& first = node.InputCount == 0 ? node : Nodes[node.Inputs[0]];

                switch (node.Code) {
                    case Input: {
                        if (node.InputCount != 0 || node.Unit != Feature) return false;

                        if (Perspectives == 0 && inputSize != 0 && inputSize != node.Size) return false;

                        inputSize = node.Size;
                        break;
                    }

                    case Affine: {
                        if (!unary() || node.Parameter >= Parameters || node.Width != first.Size ||
                            first.Unit != Feature || node.Unit != (node.FeatureAffine ? Feature : Product))
                            return false;

                        if (node.FeatureAffine && first.Code != Input) return false;

                        break;
                    }

                    case Accumulate: {
                        if (!unary() || first.Code != Affine || !first.FeatureAffine ||
                            first.Size != node.Size || node.State >= Perspectives || node.Unit != Feature)
                            return false;

                        if (stateSize != 0 && stateSize != node.Size) return false;

                        stateSize = node.Size;
                        break;
                    }

                    case AffineAccumulate: {
                        if (!unary() || first.Code != Input || first.Unit != Feature || node.Width != first.Size ||
                            node.Parameter >= Parameters || node.State >= Perspectives || node.Unit != Feature)
                            return false;

                        if (stateSize != 0 && stateSize != node.Size) return false;

                        stateSize = node.Size;
                        break;
                    }

                    case Activate: {
                        if (!unary() || first.Size != node.Size || first.Unit != node.Unit) return false;

                        if (!activationValid(node, node.Unit == Feature)) return false;

                        break;
                    }

                    case Requantize: {
                        if (!unary() || first.Unit != Product || node.Unit != Feature || first.Size != node.Size)
                            return false;

                        break;
                    }

                    case Scale: {
                        if (!unary() || first.Size != node.Size || node.Unit != Product) return false;

                        break;
                    }

                    case Add: {
                        if (node.InputCount != 2 || first.Unit != Feature || node.Unit != Feature ||
                            first.Size != node.Size || Nodes[node.Inputs[1]].Size != node.Size ||
                            Nodes[node.Inputs[1]].Unit != Feature)
                            return false;

                        break;
                    }

                    case Concat:
                    case ConcatAffine:
                    case ActivateConcatAffine: {
                        if (node.InputCount == 0) return false;

                        s00 width = 0;
                        for (s00 j = 0; j < node.InputCount; j++) {
                            const auto& input = Nodes[node.Inputs[j]];

                            if (input.Unit != Feature) return false;

                            if (input.Size > std::numeric_limits<s00>::max() - width) return false;

                            width += input.Size;
                        }

                        if (node.Code == Concat) {
                            if (node.Size != width || node.Unit != Feature) return false;
                        } else {
                            if (node.Width != width || node.Parameter >= Parameters || node.Unit != Product)
                                return false;

                            if (node.Code == ActivateConcatAffine && (
                                    node.InputCount != 2 || first.Size != Nodes[node.Inputs[1]].Size ||
                                    !activationValid(node, true)
                            )) return false;
                        }

                        break;
                    }

                    default: break;
                }

                const auto hasParameter = [](const Instruction& instruction) {
                    return instruction.Code == Affine       || instruction.Code == AffineAccumulate     ||
                           instruction.Code == ConcatAffine || instruction.Code == ActivateConcatAffine  ;
                };

                const auto featureParameter = [](const Instruction& instruction) {
                    return instruction.Code == AffineAccumulate || instruction.FeatureAffine;
                };

                if (hasParameter(node)) {
                    for (s00 prior = 0; prior < i; prior++) {
                        const auto& other = Nodes[prior];

                        if (!hasParameter(other) || other.Parameter != node.Parameter) continue;

                        if (other.Width != node.Width || other.Size != node.Size ||
                            featureParameter(other) != featureParameter(node)) return false;
                    }
                }

                if (node.Code == Accumulate || node.Code == AffineAccumulate) {
                    const auto parameter = node.Code == Accumulate ? first.Parameter : node.Parameter;

                    for (s00 prior = 0; prior < i; prior++) {
                        const auto& other = Nodes[prior];

                        if ((other.Code != Accumulate && other.Code != AffineAccumulate) || other.State != node.State)
                            continue;

                        const auto previous = other.Code == Accumulate ?
                            Nodes[other.Inputs[0]].Parameter : other.Parameter;

                        if (parameter != previous) return false;
                    }
                }

                if (node.Code == Affine && node.FeatureAffine) {
                    if (Output == i || Perspectives == 0) return false;

                    for (s00 consumer = i + 1; consumer < Count; consumer++) {
                        if (Nodes[consumer].InputCount > Capacity) return false;

                        for (s00 edge = 0; edge < Nodes[consumer].InputCount; edge++)
                            if (Nodes[consumer].Inputs[edge] == i && Nodes[consumer].Code != Accumulate) return false;
                    }
                }

                if (node.Code == Input && Perspectives != 0) {
                    if (Output == i) return false;

                    for (s00 consumer = i + 1; consumer < Count; consumer++) {
                        const auto& use = Nodes[consumer];

                        if (use.InputCount > Capacity) return false;

                        for (s00 edge = 0; edge < use.InputCount; edge++)
                            if (use.Inputs[edge] == i && use.Code != AffineAccumulate &&
                                !(use.Code == Affine && use.FeatureAffine)) return false;
                    }
                }
            }

            return true;
        }

    };

}

#endif
