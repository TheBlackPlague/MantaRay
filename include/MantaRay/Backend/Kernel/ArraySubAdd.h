//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYSUBADD_H
#define MANTARAY_BACKEND_KERNEL_ARRAYSUBADD_H

#include "../Processor.h"

namespace MantaRay::Backend::Kernel
{

    template<QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    void SubAdd(Array<T, N>& base, const Array<T, N>& sub, const Array<T, N>& add)
    {
        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

        if constexpr (HasSIMD && N >= Step && N % Step == 0) {
            for (s00 i = 0; i < N; i += Step) SIMD<T>::Store(
                SIMD<T>::Add(
                    SIMD<T>::Sub(
                        SIMD<T>::From(base, i),
                        SIMD<T>::From( sub, i)
                    ),
                    SIMD<T>::From(add, i)
                ),
                base,
                i
            );
        } else {
            for (s00 i = 0; i < N; i++) base[i] = WrapAdd(WrapSub(base[i], sub[i]), add[i]);
        }
    }

}

#endif
