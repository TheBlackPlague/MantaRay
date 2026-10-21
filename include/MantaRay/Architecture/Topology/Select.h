//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_TOPOLOGY_SELECT_H
#define MANTARAY_ARCHITECTURE_TOPOLOGY_SELECT_H

#include <string_view>

#include "../OutputBuckets.h"
#include "../Signal.h"
#include "../../Common/Message.h"

namespace MantaRay
{

    template<OutputBuckets B>
    struct Select
    {

        using Buckets = B;

        constexpr static std::string_view Name = "Select";

        consteval static Diagnostic Check(const Signal& signal)
        {
            if (signal.Perspectives != 1) return Fail(Message {} << "Select requires a single perspective");

            if (signal.Size % B::Count != 0) {
                auto message = Message {};

                message << "size " << signal.Size << " is not divisible into " << B::Count << " buckets";

                return Fail(message);
            }

            return {};
        }

        consteval static Signal Forward(const Signal& signal)
        {
            Signal output = signal;

            output.Size    = signal.Size / B::Count;
            output.Buckets = 1;

            return output;
        }

    };

}

#endif //MANTARAY_ARCHITECTURE_TOPOLOGY_SELECT_H
