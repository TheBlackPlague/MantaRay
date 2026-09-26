//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_REQUANTIZE_H
#define MANTARAY_BACKEND_KERNEL_REQUANTIZE_H

#include <algorithm>
#include <limits>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    template<typename Q>
    constexpr Q::FeatureType Requantize(const typename Q::SumType value)
    {
        using F = Q::FeatureType;

        const i64 scaled = static_cast<i64>(value) / Q::QB;

        constexpr i64 Minimum = std::numeric_limits<F>::min();
        constexpr i64 Maximum = std::numeric_limits<F>::max();

        return static_cast<F>(std::clamp(scaled, Minimum, Maximum));
    }

    template<typename Q>
    constexpr Q::SumType FinalScale(const typename Q::SumType value)
    {
        using S = Q::SumType;

        const S scaled = WrapMul(value, static_cast<S>(Q::OutputScale));
        constexpr S Divisor = static_cast<S>(i64{Q::QA} * Q::QB);

        return scaled / Divisor;
    }

}

#endif
