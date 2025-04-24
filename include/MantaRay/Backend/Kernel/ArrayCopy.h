//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYCOPY_H
#define MANTARAY_ARRAYCOPY_H

#include <cstring>

#include "../Processor.h"

namespace MantaRay
{

    template<QuantizedInteger T, s00 N>
    inline void ArrayCopy(const Array<T, N>& src, Array<T, N>& dst)
    {
#ifdef SIMD

#ifdef __ARM_NEON__

        using Vector = SIMDVEC<T>;

#else

        using Vector = SIMDVEC;

#endif

        Vector v0;

        constexpr s00 Step = sizeof(Vector) / sizeof(T);

        for (s00 i = 0; i < N; i += Step) {
            v0 = SIMD<T>::From(src, i);
            SIMD<T>::Store(v0, dst, i);
        }

#else

        std::memcpy(dst.data(), src.data(), sizeof(Array<T, N>));

#endif
    }

} // MantaRay

#endif //MANTARAY_ARRAYCOPY_H
