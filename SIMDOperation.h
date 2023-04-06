//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_SIMDOPERATION_H
#define CEREBRUM_SIMDOPERATION_H

#include <array>
#include <immintrin.h>
#include "SIMDBackend.h"

namespace Cerebrum
{

    class SIMDOperation
    {

        public:
            template<typename Type, size_t InputSize, size_t DeltaSize>
            inline static void AddToAll(std::array<Type, InputSize> &inputA, std::array<Type, InputSize> &inputB,
                                        const std::array<Type, DeltaSize> &delta,
                                        const int offsetA, const int offsetB)
            {
                for (int i = 0; i < InputSize; i++) inputA[i] += delta[offsetA + i];
                for (int i = 0; i < InputSize; i++) inputB[i] += delta[offsetB + i];
            }

            template<typename Type, size_t InputSize, size_t DeltaSize>
            inline static void SubtractFromAll(std::array<Type, InputSize> &inputA, std::array<Type, InputSize> &inputB,
                                               const std::array<Type, DeltaSize> &delta,
                                               const int offsetA, const int offsetB)
            {
                for (int i = 0; i < InputSize; i++) inputA[i] -= delta[offsetA + i];
                for (int i = 0; i < InputSize; i++) inputB[i] -= delta[offsetB + i];
            }

            template<typename Type, size_t InputSize, size_t DeltaSize>
            inline static void SubtractAndAddToAll(std::array<Type, InputSize> &inputA,
                                                   std::array<Type, InputSize> &inputB,
                                                   const std::array<Type, DeltaSize> &delta, const int offsetAS,
                                                   const int offsetAA, const int offsetBS, const int offsetBA)
            {
                for (int i = 0; i < InputSize; i++) inputA[i] -= delta[offsetAS + i];
                for (int i = 0; i < InputSize; i++) inputA[i] += delta[offsetAA + i];
                for (int i = 0; i < InputSize; i++) inputB[i] -= delta[offsetBS + i];
                for (int i = 0; i < InputSize; i++) inputB[i] += delta[offsetBA + i];
            }

            template<typename InputType, typename ActivationFunction, size_t InputSize, size_t OutputSize>
            static void ActivateFlattenAndForward(
                    const std::array<InputType, InputSize> &inputA, const std::array<InputType, InputSize> &inputB,
                    const std::array<InputType, InputSize * 2 * OutputSize> &weight,
                    const std::array<InputType, OutputSize> &bias,
                    std::array<int32_t, OutputSize> &output, const int offset
                    )
            {
                int weightStride = 0;

                for (int i = 0; i < OutputSize; i++) {
#ifdef __AVX2__
                    __m256i ymm0 = SIMDBackend<int32_t>::Zero();
                    __m256i ymm1;
                    __m256i ymm2;

                    for (int j = 0; j < InputSize; j += 16) {
                        ymm1 = SIMDBackend<InputType>::From(inputA, j);
                        ymm2 = SIMDBackend<InputType>::From(weight, weightStride + j);
                        ymm1 = ActivationFunction::Activate(ymm1);
                        ymm1 = SIMDBackend<InputType>::MultiplyAddAdjacent(ymm1, ymm2);
                        ymm0 = SIMDBackend<int32_t>::Add(ymm0, ymm1);

                        ymm1 = SIMDBackend<InputType>::From(inputB, j);
                        ymm2 = SIMDBackend<InputType>::From(weight, InputSize + weightStride + j);
                        ymm1 = ActivationFunction::Activate(ymm1);
                        ymm1 = SIMDBackend<InputType>::MultiplyAddAdjacent(ymm1, ymm2);
                        ymm0 = SIMDBackend<int32_t>::Add(ymm0, ymm1);
                    }

                    weightStride += InputSize * 2;
                    output[offset + i] = SIMDBackend<int32_t>::Sum(ymm0) + bias[offset + i];
#else
                    int32_t sum = 0;

                    for (int j = 0; j < InputSize; j++) {
                        sum += ActivationFunction::Activate(inputA[j]) * weight[weightStride + j];
                        sum += ActivationFunction::Activate(inputB[j]) * weight[InputSize + weightStride + j];
                    }

                    weightStride += InputSize * 2;
                    output[offset + i] = sum + bias[offset + i];
#endif
                }
            }

    };

} // Cerebrum

#endif //CEREBRUM_SIMDOPERATION_H
