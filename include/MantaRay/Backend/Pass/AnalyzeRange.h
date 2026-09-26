//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_PASS_ANALYZERANGE_H
#define MANTARAY_BACKEND_PASS_ANALYZERANGE_H

#include "../Graph.h"
#include "Range.h"

namespace MantaRay::Backend::Pass
{

    struct AnalyzeRange
    {

        template<typename G>
        constexpr static G Run(G graph)
        {
            using Q = G::Quantization;

            for (s00 i = 0; i < graph.Count; ++i) {
                auto& node = graph.Nodes[i];

                node.NarrowSquare = false;
                node.Bounded      = false;

                node.Lower = 0;
                node.Upper = 0;

                if (node.Code != Node::Operation::Activate             &&
                    node.Code != Node::Operation::ActivateConcatAffine  ) continue;

                if (node.Function == Node::Activation::Identity) continue;

                const i64 scale = node.Unit == Node::Domain::Feature ||
                                  node.Code == Node::Operation::ActivateConcatAffine ? Q::QA : i64{Q::QA} * Q::QB;

                const i64 lower = i64{node.Minimum} * scale;
                const i64 upper = i64{node.Maximum} * scale;

                node.Bounded = node.Code == Node::Operation::Activate;

                if (node.Bounded) {
                    node.Lower = lower;
                    node.Upper = upper;
                }

                if (node.Function == Node::Activation::Squared) {
                    node.NarrowSquare = std::is_same_v<typename Q::FeatureType, i16> &&
                                        lower >= 0 && upper <= 181 &&
                                        ProveDivision(upper * upper, scale).Valid &&
                                        (
                                            node.Unit == Node::Domain::Feature ||
                                            node.Code == Node::Operation::ActivateConcatAffine
                                        );

                    if (node.Bounded) {
                        node.Lower = lower * lower / scale;
                        node.Upper = upper * upper / scale;
                    }
                }
            }

            return graph;
        }

    };

}

#endif
