#pragma once

#include <algorithm>
#include <limits>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::Kernel
{

    template<typename Q>
    constexpr Q::FeatureType Requantize(typename Q::SumType value)
    {
        using F = Q::FeatureType;

        return static_cast<F>(
            std::clamp(
                static_cast<i64>(value) / Q::QB,
                static_cast<i64>(std::numeric_limits<F>::min()),
                static_cast<i64>(std::numeric_limits<F>::max())
            )
        );
    }

    template<typename Q>
    constexpr Q::SumType FinalScale(typename Q::SumType value)
    {
        using S = Q::SumType;

        return WrapMul(value, static_cast<S>(Q::OutputScale)) / static_cast<S>(static_cast<i64>(Q::QA) * Q::QB);
    }

}
