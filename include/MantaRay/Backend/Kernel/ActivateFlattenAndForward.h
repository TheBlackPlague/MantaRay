//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_ACTIVATEFLATTENANDFORWARD_H
#define MANTARAY_ACTIVATEFLATTENANDFORWARD_H
\
#include "../Base.h"

namespace MantaRay
{

    struct ActivationFunction {};

    template<QuantizedInteger T, T Minimum, T Maximum>
    struct ClippedReLU : ActivationFunction
    { constexpr static T Min = Minimum; constexpr static T Max = Maximum; };

    template<QuantizedInteger T, T Minimum, T Maximum>
    struct SClippedReLU : ActivationFunction
    { constexpr static T Min = Minimum; constexpr static T Max = Maximum; };

    template<typename Function, QuantizedInteger T, QuantizedInteger U, usize N, usize M>
    struct ActivateFlattenAndForwardDispatch
    {
        static Array<U, M> Dispatch(
            const Array<T, N        >& x0,
            const Array<T, N        >& x1,
            const Array<T, N * 2 * M>& w ,
            const Array<T,         M>& b ) { Array<U, M> y {}; return y; }
    };

    template<QuantizedInteger T, QuantizedInteger U, usize N, usize M, T Minimum, T Maximum>
    struct ActivateFlattenAndForwardDispatch<ClippedReLU<T, Minimum, Maximum>, T, U, N, M>
    {

        static Array<U, M> Dispatch(
            const Array<T, N        >& x0,
            const Array<T, N        >& x1,
            const Array<T, N * 2 * M>& w ,
            const Array<T,         M>& b )
        {
            HWY_ALIGN Array<U, M> y;

            constexpr LaneExpression<T> laneExprT;

            const auto v0 = Highway::Set(laneExprT, Minimum);
            const auto v1 = Highway::Set(laneExprT, Maximum);

            usize stride = 0;

            for (usize i = 0; i < M; i++) {
                constexpr LaneExpression<U> laneExprU;

                auto v2 = Highway::Zero(laneExprU);

                for (usize j = 0; j < N; j += Highway::Lanes(laneExprT)) {
                    const auto v3 = Highway::Load(laneExprT, x0.data() + j         );
                    const auto v4 = Highway::Load(laneExprT, w .data() + j + stride);

                    const auto v5 = Highway::Max(v0, v3);
                    const auto v6 = Highway::Min(v1, v5);

                    const auto v7 = Highway::WidenMulPairwiseAdd(laneExprU, v6, v4);

                    v2 = Highway::Add(v2, v7);
                }

                stride += N;

                for (usize j = 0; j < N; j += Highway::Lanes(laneExprT)) {
                    const auto v3 = Highway::Load(laneExprT, x1.data() + j         );
                    const auto v4 = Highway::Load(laneExprT, w .data() + j + stride);

                    const auto v5 = Highway::Max(v0, v3);
                    const auto v6 = Highway::Min(v1, v5);

                    const auto v7 = Highway::WidenMulPairwiseAdd(laneExprU, v6, v4);

                    v2 = Highway::Add(v2, v7);
                }

                stride += N;

                y[i] = Highway::ReduceSum(laneExprU, v2) + b[i];
            }

            return y;
        }

    };

    template<QuantizedInteger T, QuantizedInteger U, usize N, usize M, T Minimum, T Maximum>
    struct ActivateFlattenAndForwardDispatch<SClippedReLU<T, Minimum, Maximum>, T, U, N, M>
    {

        static Array<U, M> Dispatch(
            const Array<T, N        >& x0,
            const Array<T, N        >& x1,
            const Array<T, N * 2 * M>& w ,
            const Array<T,         M>& b )
        {
            HWY_ALIGN Array<U, M> y;

            constexpr LaneExpression<T> laneExprT;

            const auto v0 = Highway::Set(laneExprT, Minimum);
            const auto v1 = Highway::Set(laneExprT, Maximum);

            usize stride = 0;

            for (usize i = 0; i < M; i++) {
                constexpr LaneExpression<U> laneExprU;

                auto v2 = Highway::Zero(laneExprU);

                for (usize j = 0; j < N; j += Highway::Lanes(laneExprT)) {
                    const auto v3 = Highway::Load(laneExprT, x0.data() + j         );
                    const auto v4 = Highway::Load(laneExprT, w .data() + j + stride);

                    const auto v5 = Highway::Max(v0, v3);
                    const auto v6 = Highway::Min(v1, v5);

                    const auto v7 = Highway::Mul(laneExprT, v6, v4);
                    const auto v8 = Highway::WidenMulPairwiseAdd(laneExprU, v6, v7);

                    v2 = Highway::Add(v2, v8);
                }

                stride += N;

                for (usize j = 0; j < N; j += Highway::Lanes(laneExprT)) {
                    const auto v3 = Highway::Load(laneExprT, x1.data() + j         );
                    const auto v4 = Highway::Load(laneExprT, w .data() + j + stride);

                    const auto v5 = Highway::Max(v0, v3);
                    const auto v6 = Highway::Min(v1, v5);

                    const auto v7 = Highway::Mul(laneExprT, v6, v4);
                    const auto v8 = Highway::WidenMulPairwiseAdd(laneExprU, v6, v7);

                    v2 = Highway::Add(v2, v8);
                }

                stride += N;

                y[i] = Highway::ReduceSum(laneExprU, v2) + b[i];
            }

            return y;
        }

    };

    template<typename Function, QuantizedInteger T, QuantizedInteger U, usize N, usize M>
    [[clang::noinline]]
    Array<U, M> ActivateFlattenAndForward(
        const Array<T, N        >& x0,
        const Array<T, N        >& x1,
        const Array<T, N * 2 * M>& w ,
        const Array<T,         M>& b )
    {
        static_assert(std::is_base_of_v<ActivationFunction, Function>, "Invalid Activation Function Type");

        return ActivateFlattenAndForwardDispatch<Function, T, U, N, M>::Dispatch(x0, x1, w, b);
    }

} // MantaRay

#endif //MANTARAY_ACTIVATEFLATTENANDFORWARD_H
