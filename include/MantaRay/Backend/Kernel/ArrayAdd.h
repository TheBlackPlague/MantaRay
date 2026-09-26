//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYADD_H
#define MANTARAY_BACKEND_KERNEL_ARRAYADD_H

#include "../Processor.h"

namespace MantaRay::Backend::Kernel
{

    template<QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    void Add(Array<T, N>& base, const Array<T, N>& delta)
    {
        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

        if constexpr (HasSIMD && N >= Step && N % Step == 0) {
            for (s00 i = 0; i < N; i += Step) SIMD<T>::Store(
                SIMD<T>::Add(
                    SIMD<T>::From( base, i),
                    SIMD<T>::From(delta, i)
                ),
                base,
                i
            );
        } else {
            for (s00 i = 0; i < N; i++) base[i] = WrapAdd(base[i], delta[i]);
        }
    }

}

#endif
