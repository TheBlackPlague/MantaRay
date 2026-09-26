//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_OPTIMIZER_H
#define MANTARAY_BACKEND_OPTIMIZER_H

#include "Pass/AnalyzeRange.h"
#include "Pass/Canonicalize.h"
#include "Pass/EliminateDead.h"
#include "Pass/Fuse.h"

namespace MantaRay::Backend
{

    template<auto Source>
    struct Optimizer
    {

        static_assert(Source.Valid(), "Invalid computation graph");

        consteval static auto Optimize()
        {
            auto graph = Source;

            graph = Pass:: Canonicalize::Run(graph);
            graph = Pass::EliminateDead::Run(graph);

            for (s00 iteration = 0; iteration < Source.Count; iteration++) {
                const auto previous = graph;

                graph = Pass:: AnalyzeRange::Run(graph);
                graph = Pass::         Fuse::Run(graph);
                graph = Pass::EliminateDead::Run(graph);

                if (graph == previous) break;
            }

            graph = Pass::AnalyzeRange::Run(graph);

            return graph;
        }

        constexpr static auto Result = Optimize();

        static_assert(Result.Valid(), "Optimization produced an invalid computation graph");

    };

}

#endif
