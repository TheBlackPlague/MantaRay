//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BACKEND_KERNEL_ARRAYCOPY_H
#define MANTARAY_BACKEND_KERNEL_ARRAYCOPY_H

#include <cstring>

#include "../Processor.h"

namespace MantaRay::Backend::Kernel
{

    template<typename T, s00 N>
    [[clang::always_inline]]
    void Copy(const std::array<T, N>& src, std::array<T, N>& dst)
    {
        if constexpr (QuantizedInteger<T>) {
            constexpr s00 Step = sizeof(SIMDVEC) / sizeof(T);

            if constexpr (HasSIMD && N >= Step && N % Step == 0)
                for (s00 i = 0; i < N; i += Step) SIMD<T>::Store(SIMD<T>::From(src, i), dst, i);
            else if (&src != &dst) std::memcpy(dst.data(), src.data(), sizeof src);
        } else {
            for (s00 i = 0; i < N; i++) Copy(src[i], dst[i]);
        }
    }

}

#endif
