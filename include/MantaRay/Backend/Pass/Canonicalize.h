//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_PASS_CANONICALIZE_H
#define MANTARAY_BACKEND_PASS_CANONICALIZE_H

#include "../Graph.h"

namespace MantaRay::Backend::Pass
{

    struct Canonicalize
    {

        template<typename G>
        constexpr static G Run(G graph)
        {
            for (s00 i = 0; i < graph.Count; i++) {
                const auto& node = graph.Nodes[i];

                if (node.Code != Node::Operation::Activate || node.Function != Node::Activation::Identity) continue;

                for (s00 j = i + 1; j < graph.Count; j++) {
                    auto& consumer = graph.Nodes[j];

                    for (s00 k = 0; k < consumer.InputCount; k++)
                        if (consumer.Inputs[k] == i) consumer.Inputs[k] = node.Inputs[0];
                }

                if (graph.Output == i) graph.Output = node.Inputs[0];
            }

            return graph;
        }

    };

}

#endif
