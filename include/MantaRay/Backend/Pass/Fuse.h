//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_PASS_FUSE_H
#define MANTARAY_BACKEND_PASS_FUSE_H

#include "../Graph.h"

namespace MantaRay::Backend::Pass
{

    struct Fuse
    {

        template<typename G>
        constexpr static G Run(G graph)
        {
            using enum Node::Operation;

            const auto consumers = [&](const s00 index) {
                s00 count = graph.Output == index;

                for (s00 i = 0; i < graph.              Count; i++)
                for (s00 j = 0; j < graph.Nodes[i].InputCount; j++)
                    if (graph.Nodes[i].Inputs[j] == index) count++;

                return count;
            };

            for (s00 i = 0; i < graph.Count; i++) {
                auto& node = graph.Nodes[i];

                if (node.Code == Accumulate) {
                    const auto source = graph.Nodes[node.Inputs[0]];

                    if (source.Code != Affine || !source.FeatureAffine) continue;

                    node.Code = AffineAccumulate;

                    node.Inputs     = source.Inputs    ;
                    node.InputCount = source.InputCount;

                    node.Parameter = source.Parameter;

                    node.Width = source.Width;
                } else if (node.Code == Affine && !node.FeatureAffine) {
                    const auto source = graph.Nodes[node.Inputs[0]];

                    if (source.Code != Concat) continue;

                    node.Code = ConcatAffine;

                    node.Inputs     = source.Inputs    ;
                    node.InputCount = source.InputCount;
                } else if (node.Code == ConcatAffine && node.InputCount == 2) {
                    if (node.Inputs[0] == node.Inputs[1] ||
                        consumers(node.Inputs[0]) != 1   ||
                        consumers(node.Inputs[1]) != 1    )
                        continue;

                    const auto  first = graph.Nodes[node.Inputs[0]];
                    const auto second = graph.Nodes[node.Inputs[1]];

                    if ( first.Code    != Activate              ||
                        second.Code    != Activate              ||
                         first.Unit    != Node::Domain::Feature ||
                        second.Unit    != Node::Domain::Feature ||
                        first.Size     != second.Size           ||
                        first.Function != second.Function       ||
                        first.Minimum  != second.Minimum        ||
                        first.Maximum  != second.Maximum         )
                        continue;

                    node.Code = ActivateConcatAffine;

                    node.Inputs[0] =  first.Inputs[0];
                    node.Inputs[1] = second.Inputs[0];

                    node.Function = first.Function;

                    node.Minimum = first.Minimum;
                    node.Maximum = first.Maximum;

                    node.NarrowSquare = first.NarrowSquare;
                }
            }

            return graph;
        }

    };

}

#endif
