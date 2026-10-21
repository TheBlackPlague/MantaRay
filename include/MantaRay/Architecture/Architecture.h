//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_ARCHITECTURE_H
#define MANTARAY_ARCHITECTURE_ARCHITECTURE_H

#include <array>
#include <concepts>
#include <string_view>

#include "Signal.h"
#include "../Common/Integral.h"
#include "../Common/List.h"
#include "../Common/Message.h"

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

            Diagnostic            Diagnostic {};
            std::array<Signal, N> Signals    {};

        };

        constexpr Diagnostic Locate(const s00 index, const std::string_view name, const Diagnostic& diagnostic)
        {
            const std::string_view text { diagnostic.Text.data(), diagnostic.Text.size() };

            return Fail(Message {} << "layer " << index << " (" << name << "): " << text);
        }

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

            if constexpr (LayerCount == 0)
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

        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wc++26-extensions"
        static_assert(Result.Diagnostic.Ok, Result.Diagnostic.Text);
        #pragma clang diagnostic pop

        constexpr static std::array<Signal, LayerCount + 1> Signals = Result.Signals;

    };

}

#endif //MANTARAY_ARCHITECTURE_ARCHITECTURE_H
