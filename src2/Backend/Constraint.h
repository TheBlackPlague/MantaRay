//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CONSTRAINT_H
#define MANTARAY_CONSTRAINT_H

#include <type_traits>

#include "Integral.h"

namespace MantaRay
{

    template<typename T>
    concept QuantizedInteger = std::is_same_v<T, i08> || std::is_same_v<T, i16> || std::is_same_v<T, i32>;

} // MantaRay

#endif //MANTARAY_CONSTRAINT_H
