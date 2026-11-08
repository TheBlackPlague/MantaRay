//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ARCHITECTURE_H
#define MANTARAY_ARCHITECTURE_ARCHITECTURE_H

#include <concepts>
#include <string_view>

#include "Signal.h"
#include "../Common.h"

namespace MantaRay
{

    namespace Detail
    {

        template<typename L>
        concept Terminal = requires { requires L::Terminal; };

        template<typename L>
        concept Selecting = requires { { L::Buckets::Count } -> std::convertible_to<s00>; };

        template<s00 N>
        struct Analysis
        {

            Diagnostic       Diagnostic {};
            Array<Signal, N> Signals    {};

        };

        constexpr Diagnostic Locate(const s00 index, const std::string_view name, const Diagnostic& diagnostic)
        {
            const std::string_view text { diagnostic.Text.data(), diagnostic.Text.size() };

            return Fail(Message {} << "layer " << index << " (" << name << "): " << text);
        }

        constexpr inline s00 LineWidth = 36;

        template<s00 L>
        struct Lines
        {

            constexpr static s00 Count  = L == 0 ? 1 : (L + LineWidth - 1) / LineWidth;
            constexpr static s00 Length = (L + Count - 1) / Count;

            Array<char, Count, Length + 1> Text;

        };

        template<s00 L>
        consteval Lines<L> Split(const Message& message)
        {
            constexpr s00 Length = Lines<L>::Length;

            Lines<L> lines {};

            for (s00 line = 0; line < Lines<L>::Count; line++) {
                for (s00 i = 0; i < Length; i++) {
                    const s00 index = line * Length + i;

                    lines.Text[line][i] = index < L ? message.data()[index] : ' ';
                }

                lines.Text[line][Length] = '\0';
            }

            return lines;
        }

        template<auto>
        constexpr inline bool Rejected = false;

    }

    template<typename FT, Layer... L>
    struct Architecture
    {

        using Transformer = FT;
        using Layers      = List<L...>;

        constexpr static s00 LayerCount = sizeof...(L);

        consteval static Detail::Analysis<LayerCount + 1> Analyze()
        {
            Detail::Analysis<LayerCount + 1> analysis {};

            if (const Diagnostic diagnostic = FT::Check(); !diagnostic.Ok) {
                analysis.Diagnostic = Detail::Locate(0, FT::Name, diagnostic);

                return analysis;
            }

            analysis.Signals[0] = FT::Forward();

            if (LayerCount == 0)
                analysis.Diagnostic = Detail::Locate(0, FT::Name, Fail(Message {} << "the last layer must be Output"));

            s00 index = 0;

            ([&] {
                if (!analysis.Diagnostic.Ok) return;

                const Signal& input = analysis.Signals[index++];

                const bool terminal = Detail::Terminal<L>;
                const bool last     = index == LayerCount;

                const Diagnostic diagnostic =
                     terminal && !last ? Fail(Message {} << L::Name << " must be the last layer") :
                    !terminal &&  last ? Fail(Message {} << "the last layer must be Output")      :
                                         L::Check(input)                                          ;

                if (!diagnostic.Ok) {
                    analysis.Diagnostic = Detail::Locate(index, L::Name, diagnostic);

                    return;
                }

                analysis.Signals[index] = L::Forward(input);

                if constexpr (Detail::Selecting<L>) analysis.Signals[index - 1].Buckets = L::Buckets::Count;
            }(), ...);

            return analysis;
        }

        constexpr static Detail::Analysis<LayerCount + 1> Result = Analyze();

        constexpr static auto Report = Detail::Split<Result.Diagnostic.Text.size()>(Result.Diagnostic.Text);

        static_assert(Result.Diagnostic.Ok || Detail::Rejected<Report>, "the architecture is invalid");

        constexpr static Array<Signal, LayerCount + 1> Signals = Result.Signals;

    };

}

#endif //MANTARAY_ARCHITECTURE_ARCHITECTURE_H
