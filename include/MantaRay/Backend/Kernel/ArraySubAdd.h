#pragma once

#include "../Processor.h"

namespace MantaRay::Backend::Kernel
{

    template<QuantizedInteger T, s00 N>
    [[clang::always_inline]]
    void SubAdd(Array<T, N>& base, const Array<T, N>& sub, const Array<T, N>& add)
    {

#ifdef SIMD

        constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

        if constexpr (N >= Step && N % Step == 0) {
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
        } else

#endif

        {
            for (s00 i = 0; i < N; ++i) base[i] = WrapAdd(WrapSub(base[i], sub[i]), add[i]);
        }

    }

}
