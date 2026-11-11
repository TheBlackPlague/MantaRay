//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IR_VALIDATE_H
#define MANTARAY_IR_VALIDATE_H

#include <string_view>

#include "Analysis.h"
#include "Canonical.h"
#include "GraphIR.h"
#include "../Common.h"

namespace MantaRay::IR
{

    namespace Detail
    {

        constexpr bool Merges(Node outer, Node inner)
        {
            Node leaf {};

            leaf.Operation = Op::Parameter;
            leaf.Size      = inner.Size;
            inner.Inputs   = {};
            outer.Inputs   = { 1, 0 };

            const Array<Node, 3> nodes { leaf, inner, outer };
            const Array<Head, 1> heads { Head { .Node = 2 } };

            return Canonicalize(nodes, heads).Nodes.size() < nodes.size();
        }

        constexpr Diagnostic Uncanonical(const std::string_view what)
        { return Fail(Message {} << what << " is not canonical; call IR::Canonicalize"); }

        template<Sizes S>
        constexpr Diagnostic CheckRewrites(const Graph<S>& graph, const Node& node)
        {
            const Op first  = node.InputCount > 0 ? graph.Nodes[node.Inputs[0]].Operation : Op::None;
            const Op second = node.InputCount > 1 ? graph.Nodes[node.Inputs[1]].Operation : Op::None;

            const bool extremum = node.Operation == Op::Min      || node.Operation == Op::Max     ;
            const bool constant = first          == Op::Constant || second         == Op::Constant;

            if (node.Operation == Op::Sub) return Uncanonical("subtraction");

            if (extremum && constant) return Uncanonical("a constant operand");

            if (node.Operation == Op::Slice && first == Op::Clamp) return Uncanonical("a slice of a clamp");

            if (node.Operation == Op::Clamp && first == Op::Clamp && Merges(node, graph.Nodes[node.Inputs[0]]))
                return Uncanonical("a clamp of a clamp with overlapping bounds");

            return {};
        }

        template<Sizes S>
        constexpr s00 Probe(const Graph<S>& graph, const MantaRay::Detail::Buffer<u16>& table, const Node& node)
        {
            s00 slot = Hash(node) % table.size();

            while (table[slot] != 0 && !Same(graph.Nodes[table[slot] - 1], node)) slot = (slot + 1) % table.size();

            return slot;
        }

        template<Sizes S>
        constexpr Diagnostic CheckCanonical(const Graph<S>& graph, const Analyzer<S>& analyzer)
        {
            MantaRay::Detail::Buffer<u16> table { 2 * S.Nodes + 1 };

            for (s00 i = 0; i < S.Nodes; i++) {
                const Node& node = graph.Nodes[i];

                if (const Diagnostic diagnostic = CheckRewrites(graph, node); !diagnostic.Ok)
                    return analyzer.Locate(node, diagnostic);

                const s00 slot = Probe(graph, table, node);

                if (table[slot] != 0)
                    return analyzer.Locate(node, Fail(Message {} << "this node equals node " <<
                                                                    i64 { table[slot] - 1 } <<
                                                                    "; call IR::Canonicalize"));

                table[slot] = static_cast<u16>(i + 1);
            }

            return {};
        }

        template<Sizes S>
        constexpr Diagnostic CheckItems(const Graph<S>& graph, const Analyzer<S>& analyzer)
        {
            Array<u16, S.Tensors> saved {};

            for (s00 i = 0; i < S.Items; i++) {
                const Item& item = graph.Items[i];

                if (item.Kind != ItemKind::Tensor) continue;

                if (item.Tensor >= S.Tensors)
                    return Fail(Message {} << "file item " << static_cast<i64>(i) << " refers to tensor " <<
                                              i64 { item.Tensor } << " but the graph has " << i64 { S.Tensors } <<
                                              " tensors");

                const Tensor& tensor = graph.Tensors[item.Tensor];

                if (tensor.Role == Role::Folded)
                    return analyzer.Locate(tensor, Fail(Message {} << "a folded tensor is computed at load and " <<
                                                                      "cannot be a file item"));

                if (++saved[item.Tensor] > 1)
                    return analyzer.Locate(tensor, Explain({ .Code = FaultCode::TensorSavedTwice }));
            }

            for (s00 i = 0; i < S.Tensors; i++)
                if (saved[i] == 0 && graph.Tensors[i].Role != Role::Folded)
                    return analyzer.Locate(graph.Tensors[i], Explain({ .Code = FaultCode::TensorNotSaved }));

            return {};
        }

    }

    template<Sizes S>
    consteval Diagnostic Validate(const Graph<S>& graph)
    {
        if (const Diagnostic analyzed = Analyze(graph).Diagnostic; !analyzed.Ok) return analyzed;

        const Detail::Analyzer<S> analyzer { graph };

        if (const Diagnostic canonical = Detail::CheckCanonical(graph, analyzer); !canonical.Ok) return canonical;

        return Detail::CheckItems(graph, analyzer);
    }

}

#endif //MANTARAY_IR_VALIDATE_H
