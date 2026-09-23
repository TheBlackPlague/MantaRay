#pragma once

#include "../Processor.h"

namespace MantaRay::Backend::Kernel
{

    template<QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    void Sub(Array<T, N>& base, const Array<T, N>& delta)
    {

#ifdef SIMD

        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);
        if constexpr (N >= Step && N % Step == 0) {
            for (s00 i = 0; i < N; i += Step) SIMD<T>::Store(
                SIMD<T>::Sub(
                    SIMD<T>::From( base, i),
                    SIMD<T>::From(delta, i)
                ),
                base,
                i
            );
        } else

#endif

        {
            for (s00 i = 0; i < N; ++i) base[i] = WrapSub(base[i], delta[i]);
        }

    }

}
