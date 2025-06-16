//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ARRAYOPERATE_H
#define MANTARAY_ARRAYOPERATE_H

#include "../Base.h"

namespace MantaRay
{

    enum Operation : u08 { Add, Sub };

    template<Operation Op, QuantizedInteger T, usize N>
    using Operand = Array<T, N>;

    template<Operation... Operations, QuantizedInteger T, usize N>
    [[clang::always_inline]]
    void ArrayOperate(Array<T, N>& base, const Operand<Operations, T, N>&... operands)
    {
        static_assert(sizeof...(Operations) == sizeof...(operands),
            "Not enough operands for the specified operations.");

        constexpr LaneExpression<T> laneExpr;

        for (usize i = 0; i < N; i += Highway::Lanes(laneExpr)) {
            auto v0 = Highway::Load(laneExpr, base.data() + i);

            const auto DoOp = [&](const auto Op, const Array<T, N>& operand)
            {
                const auto v1 = Highway::Load(laneExpr, operand.data() + i);

                if constexpr (Op == Add) v0 = Highway::Add(v0, v1);
                if constexpr (Op == Sub) v0 = Highway::Sub(v0, v1);
            };

            (DoOp(std::integral_constant<Operation, Operations>(), operands), ...);

            Highway::Store(v0, laneExpr, base.data() + i);
        }
    }

} // MantaRay

#endif //MANTARAY_ARRAYOPERATE_H
