//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_NNUE_H
#define CEREBRUM_NNUE_H

#include <array>
#include <cstdint>

#include "Accumulator.h"
#include "SIMDOperation.h"

namespace Cerebrum
{

    template<typename BaseType, typename ActivationFunction,
            size_t InputSize, size_t HiddenSize, size_t OutputSize, size_t AccumulatorStackSize,
            uint16_t Scale, uint16_t QuantizationFeature, uint16_t QuantizationOutput>
    class PerspectiveNetwork
    {

        private:
            std::array<BaseType, InputSize * HiddenSize> FeatureWeight;
            std::array<BaseType, HiddenSize> FeatureBias;
            std::array<BaseType, HiddenSize * 2 * OutputSize> OutputWeight;
            std::array<BaseType, OutputSize> OutputBias;
            std::array<int32_t, OutputSize> Output;

            std::array<Accumulator<BaseType, HiddenSize>, AccumulatorStackSize> Accumulators;
            uint16_t CurrentAccumulator;

            const uint16_t ColorStride = 64 * 6;
            const uint8_t PieceStride = 64;

        public:
            PerspectiveNetwork()
            {
                Accumulator<BaseType, HiddenSize> accumulator;
                std::fill(std::begin(Accumulators), std::end(Accumulators), accumulator);
                CurrentAccumulator = 0;
            }

            inline void ResetAccumulator()
            {
                CurrentAccumulator = 0;
            }

            inline void PushAccumulator()
            {
                Accumulators[CurrentAccumulator].CopyTo(Accumulators[++CurrentAccumulator]);
            }

            inline void PullAccumulator()
            {
                static_assert(CurrentAccumulator > 0, "Calling PullAccumulator() with CurrentAccumulator = 0.");

                CurrentAccumulator--;
            }

            inline void RefreshAccumulator()
            {
                Accumulator<BaseType, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];
                accumulator.Zero();
                accumulator.LoadBias(FeatureBias);
            }

            inline void EfficientlyUpdateAccumulator(const uint8_t piece, const uint8_t color,
                                                     const uint8_t from, const uint8_t to)
            {
                const uint16_t pieceStride = piece * PieceStride;

                const uint32_t whiteIndexFrom = color * ColorStride + pieceStride + from;
                const uint32_t blackIndexFrom = (color ^ 1) * ColorStride + pieceStride + (from ^ 56);
                const uint32_t whiteIndexTo = color * ColorStride + pieceStride + to;
                const uint32_t blackIndexTo = (color ^ 1) * ColorStride + pieceStride + (to ^ 56);

                Accumulator<BaseType, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

                SIMDOperation::SubtractAndAddToAll(accumulator.White, accumulator.Black, FeatureWeight,
                                                   whiteIndexFrom * HiddenSize,
                                                   whiteIndexTo * HiddenSize,
                                                   blackIndexFrom * HiddenSize,
                                                   blackIndexTo * HiddenSize);
            }

            template<AccumulatorOperation Operation>
            inline void EfficientlyUpdateAccumulator(const uint8_t piece, const uint8_t color, const uint8_t sq)
            {
                const uint16_t pieceStride = piece * PieceStride;

                const uint32_t whiteIndex = color * ColorStride + pieceStride + sq;
                const uint32_t blackIndex = (color ^ 1) * ColorStride + pieceStride + (sq ^ 56);

                Accumulator<BaseType, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

                if (Operation == AccumulatorOperation::Activate)
                    SIMDOperation::AddToAll(accumulator.White, accumulator.Black,FeatureWeight,
                                            whiteIndex * HiddenSize, blackIndex * HiddenSize);

                else SIMDOperation::SubtractFromAll(accumulator.White, accumulator.Black, FeatureWeight,
                                                    whiteIndex * HiddenSize, blackIndex * HiddenSize);
            }

            inline int32_t Evaluate(const uint8_t colorToMove)
            {
                Accumulator<BaseType, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

                if (colorToMove == 0) SIMDOperation::ActivateFlattenAndForward<BaseType, ActivationFunction>(
                            accumulator.White, accumulator.Black,
                            OutputWeight, OutputBias, Output, 0
                    );
                else SIMDOperation::ActivateFlattenAndForward<BaseType, ActivationFunction>(
                            accumulator.Black, accumulator.White,
                            OutputWeight, OutputBias, Output, 0
                    );

                return Output[0] * Scale / (QuantizationFeature * QuantizationOutput);
            }

    };

} // Cerebrum

#endif //CEREBRUM_NNUE_H
