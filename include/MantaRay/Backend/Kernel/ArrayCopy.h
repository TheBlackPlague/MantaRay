//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYCOPY_H
#define MANTARAY_ARRAYCOPY_H

#include "../Base.h"

namespace MantaRay
{

    template<QuantizedInteger T, usize N>
    [[clang::always_inline]]
    void ArrayCopy(const Array<T, N>& src, Array<T, N>& dst)
    {
        constexpr LaneExpression<T> laneExpr;

        for (usize i = 0; i < N; i += Highway::Lanes(laneExpr))
            Highway::Store(Highway::Load(laneExpr, src.data() + i), laneExpr, dst.data() + i);
    }

} // MantaRay

#endif //MANTARAY_ARRAYCOPY_H
