//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IR_DUMP_H
#define MANTARAY_IR_DUMP_H

#include <bit>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "Analysis.h"
#include "GraphIR.h"
#include "../Common.h"

namespace MantaRay::IR
{

    namespace Detail
    {

        constexpr inline Array<std::string_view, 19> Functions {
            "None",
            "Abs", "Sign", "Exp", "Log", "Sqrt", "Reciprocal", "Sigmoid", "Tanh",
            "Sin", "Cos", "Tan", "Sinh", "Cosh", "Round", "Truncate", "IsPositive", "IsZero", "IsNonNegative"
        };

        constexpr inline Array<std::string_view, 8> Types { "None", "I08", "U08", "I16", "U16", "I32", "I64", "F32" };

        constexpr inline Array<std::string_view, 4> Bounds       { "None", "Low", "High", "Both" };
        constexpr inline Array<std::string_view, 4> Perspectives { "None", "Stm", "Ntm", "Shared" };
        constexpr inline Array<std::string_view, 3> Reductions   { "Sequential", "BiasFirstFma", "Lanes16" };

        constexpr inline Array<std::string_view, 5> Domains     { "None", "Sparse", "Integer", "Float", "Constant" };
        constexpr inline Array<std::string_view, 4> Provenances { "Proven", "LoadVerified", "Assumed", "Unbounded" };
        constexpr inline Array<std::string_view, 5> Dependences { "None", "Shared", "Stm", "Ntm", "Both" };

        static_assert(
            Functions.size() == static_cast<s00>(Function::IsNonNegative) + 1,
            "Functions must list every Function"
        );

        static_assert(Types.size() == static_cast<s00>(DType::F32) + 1, "Types must list every DType");

        static_assert(Bounds.size() == static_cast<s00>(Bound::Both) + 1, "Bounds must list every Bound");

        static_assert(
            Perspectives.size() == static_cast<s00>(Perspective::Shared) + 1,
            "Perspectives must list every Perspective"
        );

        static_assert(
            Reductions.size() == static_cast<s00>(Reduction::Lanes16) + 1,
            "Reductions must list every Reduction"
        );

        static_assert(Domains.size() == static_cast<s00>(Domain::Constant) + 1, "Domains must list every Domain");

        static_assert(
            Provenances.size() == static_cast<s00>(Provenance::Unbounded) + 1,
            "Provenances must list every Provenance"
        );

        static_assert(
            Dependences.size() == static_cast<s00>(Dependence::Both) + 1,
            "Dependences must list every Dependence"
        );

        struct Real
        {

            f64 Value = 0;

        };

        struct Hex
        {

            u64 Value = 0;

        };

        class Writer
        {

            std::string Text {};

            constexpr void Nibbles(const u64 value, s00 count)
            {
                constexpr std::string_view Digits = "0123456789abcdef";

                while (count != 0) Text.push_back(Digits[value >> (4 * --count) & 0xF]);
            }

            public:
            constexpr Writer& operator <<(const std::string_view text)
            {
                Text.append(text);

                return *this;
            }

            constexpr Writer& operator <<(const i64 value)
            {
                u64 rest = value < 0 ? 0 - static_cast<u64>(value) : static_cast<u64>(value);

                Array<char, 20> digits {};

                s00 count = 0;

                do {
                    digits[count++] = static_cast<char>('0' + rest % 10);

                    rest /= 10;
                } while (rest != 0);

                if (value < 0) Text.push_back('-');

                while (count != 0) Text.push_back(digits[--count]);

                return *this;
            }

            constexpr Writer& operator <<(const Rational value)
            {
                *this << value.Numerator;

                if (value.Denominator != 1) *this << "/" << value.Denominator;

                return *this;
            }

            constexpr Writer& operator <<(const Hex hex)
            {
                s00 count = 1;

                while (count < 16 && hex.Value >> (4 * count) != 0) count++;

                Nibbles(hex.Value, count);

                return *this;
            }

            constexpr Writer& operator <<(const Real real)
            {
                constexpr f64 Exact = 9007199254740992.0;

                const f64 value = real.Value;

                if (value >=  Unbounded) return *this << "inf";
                if (value <= -Unbounded) return *this << "-inf";

                if (value > -Exact && value < Exact && value == static_cast<f64>(static_cast<i64>(value)))
                    return *this << static_cast<i64>(value);

                const u64 bits     = std::bit_cast<u64>(value);
                const u64 mantissa = bits & 0xFFFFFFFFFFFFF;
                const i64 exponent = static_cast<i64>(bits >> 52 & 0x7FF);

                u64 fraction = mantissa;
                s00 digits   = mantissa == 0 ? 0 : 13;

                while (digits != 0 && (fraction & 0xF) == 0) {
                    fraction >>= 4;
                    digits--;
                }

                *this << (bits >> 63 ? "-" : "") << (exponent == 0 ? "0x0" : "0x1") << (digits == 0 ? "" : ".");

                Nibbles(fraction, digits);

                return *this << "p" << (exponent == 0 ? i64 { -1022 } : exponent - 1023);
            }

            constexpr Writer& Field(const std::string_view key, const i64 value)
            {
                if (value != 0) *this << " " << key << "=" << value;

                return *this;
            }

            template<typename E>
            constexpr Writer& Field(
                const std::string_view                  key  ,
                const std::span<const std::string_view> names,
                const E                                 value
            )
            {
                const s00 index = static_cast<s00>(value);

                if (index != 0) *this << " " << key << (key.empty() ? "" : "=") << names[index];

                return *this;
            }

            constexpr std::string Release() { return std::move(Text); }

        };

        template<Sizes S>
        constexpr void WriteNode(Writer& out, const Graph<S>& graph, const s00 index)
        {
            const Node& node = graph.Nodes[index];

            out << "%" << static_cast<i64>(index) << " " << Name(node.Operation);

            for (s00 k = 0; k < node.InputCount; k++) out << " %" << i64 { node.Inputs[k] };

            out << " [" << i64 { node.Size } << "]";

            out.Field("bounds"     , Bounds      , node.Bounds     );
            out.Field("function"   , Functions   , node.Function   );
            out.Field("perspective", Perspectives, node.Perspective);
            out.Field("type"       , Types       , node.Type       );
            out.Field("reduction"  , Reductions  , node.Reduction  );

            out.Field("tensor" , node.Tensor );
            out.Field("binding", node.Binding);
            out.Field("shift"  , node.Shift  );
            out.Field("begin"  , node.Begin  );
            out.Field("end"    , node.End    );
            out.Field("offset" , node.Offset );

            if (Defined(node.Low )) out << " low="  << node.Low ;
            if (Defined(node.High)) out << " high=" << node.High;
        }

        constexpr void WriteFact(Writer& out, const Fact& fact)
        {
            out << " |";

            out.Field("", Domains    , fact.Domain     );
            out.Field("", Dependences, fact.Perspective);

            if (Defined(fact.Scale)) out << " scale=" << fact.Scale;

            if (fact.Range.Min != 0 || fact.Range.Max != 0)
                out << " range=[" << fact.Range.Min << "," << fact.Range.Max << "]";

            if (fact.Low != 0 || fact.High != 0)
                out << " real=[" << Real { fact.Low } << "," << Real { fact.High } << "]";

            out.Field("", Types      , fact.Storage   );
            out.Field("", Provenances, fact.Provenance);

            if (fact.Linear     ) out << " linear";
            if (fact.FeatureSets) out << " featuresets=0x" << Hex { fact.FeatureSets };

            out.Field("selector" , fact.Selector );
            out.Field("consumers", fact.Consumers);
        }

        template<Sizes S>
        constexpr void WriteTail(Writer& out, const Graph<S>& graph, const s00 index)
        {
            const Node&            node = graph.Nodes[index];
            const std::string_view file = FileOf(graph, node.File);

            for (s00 h = 0; h < S.Heads; h++)
                if (graph.Heads[h].Node == index) out << " head=" << static_cast<i64>(h) << ":" << graph.Heads[h].Scale;

            out << " @" << file << (file.empty() ? "" : ":") << i64 { node.Line } << ":" << i64 { node.Column } << "\n";
        }

    }

    template<Sizes S>
    constexpr std::string Dump(const Graph<S>& graph)
    {
        Detail::Writer out {};

        for (s00 i = 0; i < S.Nodes; i++) {
            Detail::WriteNode(out, graph, i);
            Detail::WriteTail(out, graph, i);
        }

        return out.Release();
    }

    template<Sizes S, s00 N = S.Nodes>
    constexpr std::string Dump(const Graph<S>& graph, const Array<Fact, N>& facts)
    {
        Detail::Writer out {};

        for (s00 i = 0; i < S.Nodes; i++) {
            Detail::WriteNode(out, graph, i);
            Detail::WriteFact(out, facts[i]);
            Detail::WriteTail(out, graph, i);
        }

        return out.Release();
    }

}

#endif //MANTARAY_IR_DUMP_H
