//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TOPOLOGY_CONCAT_H
#define MANTARAY_ARCHITECTURE_TOPOLOGY_CONCAT_H

#include <string_view>

#include "../Signal.h"
#include "../../Common.h"

namespace MantaRay
{

    struct Concat
    {

        constexpr static std::string_view Name = "Concat";

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (signal.Perspectives != 2) return Fail(Message {} << "Concat requires two perspectives");

            return {};
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = signal;

            output.Size         = signal.Size * 2;
            output.Perspectives = 1;

            return output;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_TOPOLOGY_CONCAT_H
