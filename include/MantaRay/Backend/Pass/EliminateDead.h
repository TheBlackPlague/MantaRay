//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_PASS_ELIMINATEDEAD_H
#define MANTARAY_BACKEND_PASS_ELIMINATEDEAD_H

#include "../Graph.h"

namespace MantaRay::Backend::Pass
{

    struct EliminateDead
    {

        template<typename G>
        constexpr static G Run(const G& graph)
        {
            std::array<bool, G::Limit> live {};

            live[graph.Output] = true;

            for (s00 i = graph.Count; i-- > 0;) {
                if (!live[i]) continue;

                for (s00 j = 0; j < graph.Nodes[i].InputCount; j++) live[graph.Nodes[i].Inputs[j]] = true;
            }

            G result = graph;

            result.Count = 0;

            std::array<s00, G::Limit> remap {};

            for (s00 i = 0; i < graph.Count; i++) {
                if (!live[i]) continue;

                auto node = graph.Nodes[i];
                for (s00 j = 0; j < node.InputCount; j++) node.Inputs[j] = remap[node.Inputs[j]];

                remap[i] = result.Append(node);
            }

            result.Output = remap[graph.Output];

            for (s00 i = result.Count; i < G::Limit; i++) result.Nodes[i] = {};

            return result;
        }

    };

}

#endif
