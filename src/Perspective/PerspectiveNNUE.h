//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_PERSPECTIVENNUE_H
#define CEREBRUM_PERSPECTIVENNUE_H

#include <array>
#include <cstdint>
#include <cassert>

#include "PerspectiveAccumulator.h"
#include "../SIMD.h"
#include "../AccumulatorOperation.h"

namespace Cerebrum
{

    template<typename T, typename OT, typename Activation, uint16_t InputSize, uint16_t HiddenSize, uint16_t OutputSize,
            uint16_t AccumulatorStackSize, uint16_t Scale, uint16_t QuantizationFeature, uint16_t QuantizationOutput>
    class PerspectiveNetwork
    {

    private:
        std::array<T, InputSize * HiddenSize> FeatureWeight;
        std::array<T, HiddenSize> FeatureBias;
        std::array<T, HiddenSize * 2 * OutputSize> OutputWeight;
        std::array<T, OutputSize> OutputBias;
        std::array<OT, OutputSize> Output;

        std::array<PerspectiveAccumulator<T, HiddenSize>, AccumulatorStackSize> Accumulators;
        uint16_t CurrentAccumulator;

        const uint16_t ColorStride = 64 * 6;
        const uint8_t PieceStride = 64;

    public:
        PerspectiveNetwork()
        {
            PerspectiveAccumulator<T, HiddenSize> accumulator;
            std::fill(std::begin(Accumulators), std::end(Accumulators), accumulator);
            CurrentAccumulator = 0;
        }

        __attribute__((unused)) inline void ResetAccumulator()
        {
            CurrentAccumulator = 0;
        }

        __attribute__((unused)) inline void PushAccumulator()
        {
            Accumulators[CurrentAccumulator].CopyTo(Accumulators[++CurrentAccumulator]);
        }

        __attribute__((unused)) inline void PullAccumulator()
        {
            assert(CurrentAccumulator > 0);

            CurrentAccumulator--;
        }

        __attribute__((unused)) inline void RefreshAccumulator()
        {
            PerspectiveAccumulator<T, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];
            accumulator.Zero();
            accumulator.LoadBias(FeatureBias);
        }

        __attribute__((unused)) inline void EfficientlyUpdateAccumulator(const uint8_t piece, const uint8_t color,
                                                                         const uint8_t from, const uint8_t to)
        {
            const uint16_t pieceStride = piece * PieceStride;

            const uint32_t whiteIndexFrom = color * ColorStride + pieceStride + from;
            const uint32_t blackIndexFrom = (color ^ 1) * ColorStride + pieceStride + (from ^ 56);
            const uint32_t whiteIndexTo = color * ColorStride + pieceStride + to;
            const uint32_t blackIndexTo = (color ^ 1) * ColorStride + pieceStride + (to ^ 56);

            PerspectiveAccumulator<T, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

            SIMD::SubtractAndAddToAll(accumulator.White, accumulator.Black,
                                      FeatureWeight,
                                      whiteIndexFrom * HiddenSize,
                                      whiteIndexTo * HiddenSize,
                                      blackIndexFrom * HiddenSize,
                                      blackIndexTo * HiddenSize);
        }

        template<AccumulatorOperation Operation>
        __attribute__((unused)) inline void EfficientlyUpdateAccumulator(const uint8_t piece, const uint8_t color,
                                                                         const uint8_t sq)
        {
            const uint16_t pieceStride = piece * PieceStride;

            const uint32_t whiteIndex = color * ColorStride + pieceStride + sq;
            const uint32_t blackIndex = (color ^ 1) * ColorStride + pieceStride + (sq ^ 56);

            PerspectiveAccumulator<T, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

            if (Operation == AccumulatorOperation::Activate)
                SIMD::AddToAll(accumulator.White, accumulator.Black,
                               FeatureWeight,
                               whiteIndex * HiddenSize, blackIndex * HiddenSize);

            else SIMD::SubtractFromAll(accumulator.White, accumulator.Black, FeatureWeight,
                                       whiteIndex * HiddenSize, blackIndex * HiddenSize);
        }

        __attribute__((unused)) inline int32_t Evaluate(const uint8_t colorToMove)
        {
            PerspectiveAccumulator<T, HiddenSize>& accumulator = Accumulators[CurrentAccumulator];

            if (colorToMove == 0)
                SIMD::ActivateFlattenAndForward<Activation>(
                        accumulator.White, accumulator.Black, OutputWeight, OutputBias,
                        Output, 0);
            else SIMD::ActivateFlattenAndForward<Activation>(
                        accumulator.Black, accumulator.White, OutputWeight, OutputBias,
                        Output, 0);

            return Output[0] * Scale / (QuantizationFeature * QuantizationOutput);
        }

    };

} // Cerebrum

#endif //CEREBRUM_PERSPECTIVENNUE_H
