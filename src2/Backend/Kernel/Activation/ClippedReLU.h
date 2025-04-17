//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_CLIPPEDRELU_H
#define MANTARAY_CLIPPEDRELU_H

#include "../../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, typename V>
    concept ValidClippedReLUArg =
#ifdef SIMD
    (

#ifdef __ARM_NEON__

        std::is_same_v<V, SIMDVEC<T>>

#else

        std::is_same_v<V, SIMDVEC>

#endif

    );
#else

        std::is_same_v<V,         T >;

#endif

    template<QuantizedInteger T, typename V, V Minimum, V Maximum>
    inline V ClippedReLU(const V arg) requires ValidClippedReLUArg<T, V>
    {
#ifdef SIMD

        return SIMD<T>::Max(Minimum, SIMD<T>::Min(Maximum, arg));

#else

        return std::max(Minimum, std::min(Maximum, arg));

#endif
    }

}

#endif //MANTARAY_CLIPPEDRELU_H
