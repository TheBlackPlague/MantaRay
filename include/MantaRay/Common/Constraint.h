//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_CONSTRAINT_H
#define MANTARAY_COMMON_CONSTRAINT_H

#include <concepts>

#include "Integral.h"

namespace MantaRay
{

    template<typename T>
    concept QuantizedInteger = std::same_as<T, i08> || std::same_as<T, i16> || std::same_as<T, i32>;

}

#endif
