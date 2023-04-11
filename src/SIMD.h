//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_SIMD_H
#define CEREBRUM_SIMD_H

#include <array>

#include "Backend/Avx2.h"

namespace Cerebrum
{

    class SIMD
    {
        public:
            template<typename T, size_t InputSize, size_t DeltaSize>
            static inline void AddToAll(std::array<T, InputSize> &inputA, std::array<T, InputSize> &inputB,
                                        const std::array<T, DeltaSize> &delta,
                                        const uint32_t oA, const uint32_t oB)
            {
#ifdef __AVX2__
                Vec256I ymm0;
                Vec256I ymm1;

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputA, i);
                    ymm1 = Avx<T>::From(delta, oA + i);
                    ymm0 = Avx2<T>::Add(ymm0, ymm1);
                    Avx<T>::Store(ymm0, inputA, i);
                }

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputB, i);
                    ymm1 = Avx<T>::From(delta, oB + i);
                    ymm0 = Avx2<T>::Add(ymm0, ymm1);
                    Avx<T>::Store(ymm0, inputB, i);
                }
#else
                for (size_t i = 0; i < InputSize; i++) inputA[i] += delta[oA + i];
                for (size_t i = 0; i < InputSize; i++) inputB[i] += delta[oB + i];
#endif
            }

            template<typename T, size_t InputSize, size_t DeltaSize>
            static inline void SubtractFromAll(std::array<T, InputSize> &inputA, std::array<T, InputSize> &inputB,
                                               const std::array<T, DeltaSize> &delta,
                                               const uint32_t oA, const uint32_t oB)
            {
#ifdef __AVX2__
                Vec256I ymm0;
                Vec256I ymm1;

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputA, i);
                    ymm1 = Avx<T>::From(delta, oA + i);
                    ymm0 = Avx2<T>::Subtract(ymm0, ymm1);
                    Avx<T>::Store(ymm0, inputA, i);
                }

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputB, i);
                    ymm1 = Avx<T>::From(delta, oB + i);
                    ymm0 = Avx2<T>::Subtract(ymm0, ymm1);
                    Avx<T>::Store(ymm0, inputB, i);
                }
#else
                for (size_t i = 0; i < InputSize; i++) inputA[i] -= delta[oA + i];
                for (size_t i = 0; i < InputSize; i++) inputB[i] -= delta[oB + i];
#endif
            }

            template<typename T, size_t InputSize, size_t DeltaSize>
            static inline void SubtractAndAddToAll(std::array<T, InputSize> &inputA, std::array<T, InputSize> &inputB,
                                                   const std::array<T, DeltaSize> &delta,
                                                   const uint32_t oAS, const uint32_t oAA,
                                                   const uint32_t oBS, const uint32_t oBA)
            {
#ifdef __AVX2__
                Vec256I ymm0;
                Vec256I ymm1;
                Vec256I ymm2;

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputA, i);
                    ymm1 = Avx<T>::From(delta, oAS + i);
                    ymm2 = Avx<T>::From(delta, oAA + i);
                    ymm0 = Avx2<T>::Subtract(ymm0, ymm1);
                    ymm0 = Avx2<T>::Add(ymm0, ymm2);
                    Avx<T>::Store(ymm0, inputA, i);
                }

                for (size_t i = 0; i < InputSize; i += 16) {
                    ymm0 = Avx<T>::From(inputB, i);
                    ymm1 = Avx<T>::From(delta, oBS + i);
                    ymm2 = Avx<T>::From(delta, oBA + i);
                    ymm0 = Avx2<T>::Subtract(ymm0, ymm1);
                    ymm0 = Avx2<T>::Add(ymm0, ymm2);
                    Avx<T>::Store(ymm0, inputB, i);
                }
#else
                for (size_t i = 0; i < InputSize; i++) {
                    inputA[i] = inputA[i] - delta[oAS + i] + delta[oAA + i];
                    inputB[i] = inputB[i] - delta[oBS + i] + delta[oBA + i];
                }
#endif
            }

            template<typename Activation, typename T, typename OT, size_t InputSize, size_t OutputSize>
            static void ActivateFlattenAndForward(
                    const std::array<T, InputSize> &inputA, const std::array<T, InputSize> &inputB,
                    const std::array<T, InputSize * 2 * OutputSize> &weight,
                    const std::array<T, OutputSize> &bias,
                    std::array<OT, OutputSize> &output, const uint32_t o)
            {
                int stride = 0;
                for (size_t i = 0; i < OutputSize; i++) {
#ifdef __AVX2__
                    Vec256I ymm0 = Avx<OT>::Zero();
                    Vec256I ymm1;
                    Vec256I ymm2;

                    for (size_t j = 0; j < InputSize; j += 16) {
                        // START INPUT A
                        ymm1 = Avx<T>::From(inputA, j);
                        ymm2 = Avx<T>::From(weight, stride + j);
                        ymm1 = Activation::Activate(ymm1);
                        ymm1 = Avx2<T>::MultiplyAndAddAdjacent(ymm1, ymm2);
                        ymm0 = Avx2<OT>::Add(ymm0, ymm1);
                        // END INPUT A

                        // START INPUT B
                        ymm1 = Avx<T>::From(inputB, j);
                        ymm2 = Avx<T>::From(weight, InputSize + stride + j);
                        ymm1 = Activation::Activate(ymm1);
                        ymm1 = Avx2<T>::MultiplyAndAddAdjacent(ymm1, ymm2);
                        ymm0 = Avx2<OT>::Add(ymm0, ymm1);
                        // END INPUT B
                    }

                    stride += InputSize * 2;

                    output[o + i] = Avx2<OT>::Sum(ymm0) + bias[o + i];
#else
                    OT sum = 0;

                    for (size_t j = 0; j < InputSize; j++) {
                        sum += Activation::Activate(inputA[j]) * weight[stride + j];
                        sum += Activation::Activate(inputB[j]) * weight[InputSize + stride + j];
                    }

                    stride += InputSize * 2;
                    output[o + i] = sum + bias[o + i];
#endif
                }
            }

    };
} // Cerebrum

#endif //CEREBRUM_SIMD_H
