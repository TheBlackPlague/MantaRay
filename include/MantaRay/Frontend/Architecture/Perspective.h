//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_PERSPECTIVE_H
#define MANTARAY_PERSPECTIVE_H

#include <sstream>
#include <string>

#include "../../Backend/Kernel/ActivateFlattenAndForward.h"
#include "../../Backend/Kernel/ArrayAdd.h"
#include "../../Backend/Kernel/ArraySub.h"
#include "../../Backend/Kernel/ArraySubAdd.h"

#include "../IO/BinaryFileStream.h"
#include "../IO/BinaryMemoryStream.h"

#include "Accumulator.h"

namespace MantaRay
{

    template<QuantizedInteger I,
             QuantizedInteger O,
             auto ActivationFunction,
             s00  InputSize,
             s00 HiddenSize,
             s00 OutputSize,
             s00 AccumulatorStackSize,
             I Scale,
             I QuantizationFeature,
             I QuantizationOutput>
    class Perspective
    {

        static_assert(AccumulatorStackSize > 0, "The accumulator stack size must at least be greater than zero.");

        static_assert(Scale > 0 && QuantizationFeature > 127 && QuantizationOutput > 31,
                      "These scale and quantization constants don't seem right.");

        constexpr static s00 ColorStride = 64 * 6;
        constexpr static s00 PieceStride = 64    ;

        ALIGN Array<I,  InputSize *     HiddenSize> FeatureWeight;
        ALIGN Array<I,                  HiddenSize> FeatureBias  ;
        ALIGN Array<I, HiddenSize * 2 * OutputSize> OutputWeight ;
        ALIGN Array<I,                  OutputSize> OutputBias   ;

        using Accumulator      =            Accumulator<I,         HiddenSize>;
        using AccumulatorStack = std::array<Accumulator, AccumulatorStackSize>;

        AccumulatorStack Accumulators;
        s00              AccumulatorP;

        void Initialize()
        {
            const Accumulator accumulator;
            std::fill(std::begin(Accumulators), std::end(Accumulators), accumulator);

            AccumulatorP = 0;
        }

        public:
        Perspective() { Initialize(); } // NOLINT(*-pro-type-member-init)

        Perspective(BinaryFileStream<>& stream)
        {
            Initialize();

            stream.ReadArray(FeatureWeight);
            stream.ReadArray(FeatureBias  );
            stream.ReadArray( OutputWeight);
            stream.ReadArray( OutputBias  );
        }

        Perspective(BinaryMemoryStream& stream)
        {
            Initialize();

            stream.ReadArray(FeatureWeight);
            stream.ReadArray(FeatureBias  );
            stream.ReadArray( OutputWeight);
            stream.ReadArray( OutputBias  );
        }

        static std::string Info()
        {
            std::stringstream ss;

            ss << "(" << InputSize << "->" << HiddenSize << ")" << "x2" << "->" << OutputSize << std::endl;

            ss << "Details:" << std::endl;
            ss << " | " << "First  Layer Size    : " <<  InputSize                  << std::endl;
            ss << " | " << "Hidden Layer Size    : " <<                  HiddenSize << std::endl;
            ss << " | " << "Output Layer Size    : " <<                  OutputSize << std::endl;
            ss << " | " << "Input ->Hidden Weight: " <<  InputSize *     HiddenSize << std::endl;
            ss << " | " << "Hidden->Output Weight: " << HiddenSize * 2 * OutputSize << std::endl;
            ss << " | " << "AccumulatorStackSize : " <<        AccumulatorStackSize << std::endl;
            ss << " | " << "Scale                : " <<                       Scale << std::endl;
            ss << " | " << "QuantizationFeature  : " <<         QuantizationFeature << std::endl;
            ss << " | " << "QuantizationOutput   : " <<         QuantizationOutput  << std::endl;

            return ss.str();
        }

        inline void Reset() { AccumulatorP = 0; }

        inline void Push()
        {
            Accumulators[++AccumulatorP] = Accumulators[AccumulatorP - 1];

            assert(AccumulatorP < AccumulatorStackSize);
        }

        inline void Pop()
        {
            assert(AccumulatorP > 0);

            AccumulatorP--;
        }

        inline void Refresh()
        {
            Accumulators[AccumulatorP].Zero();
            Accumulators[AccumulatorP].Bias(FeatureBias);
        }

        inline void Move(const u08 piece, const u08 color, const u08 from, const u08 to)
        {
            const s00 fromIdxV =  color      * ColorStride + piece * PieceStride +  from      ;
            const s00 fromIdxU = (color ^ 1) * ColorStride + piece * PieceStride + (from ^ 56);
            const s00   toIdxV =  color      * ColorStride + piece * PieceStride +  to        ;
            const s00   toIdxU = (color ^ 1) * ColorStride + piece * PieceStride + (to   ^ 56);

            Accumulator& accumulator = Accumulators[AccumulatorP];

            ArraySubAdd(
                accumulator[0],
                Slice<HiddenSize>(FeatureWeight, fromIdxV * HiddenSize),
                Slice<HiddenSize>(FeatureWeight,   toIdxV * HiddenSize)
            );
            ArraySubAdd(
                accumulator[1],
                Slice<HiddenSize>(FeatureWeight, fromIdxU * HiddenSize),
                Slice<HiddenSize>(FeatureWeight,   toIdxU * HiddenSize)
            );
        }

        inline void Insert(const u08 piece, const u08 color, const u08 sq)
        {
            const s00 vIdx =  color      * ColorStride + piece * PieceStride +  sq      ;
            const s00 uIdx = (color ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            Accumulator& accumulator = Accumulators[AccumulatorP];

            ArrayAdd(
                accumulator[0],
                Slice<HiddenSize>(FeatureWeight, vIdx * HiddenSize)
            );
            ArrayAdd(
                accumulator[1],
                Slice<HiddenSize>(FeatureWeight, uIdx * HiddenSize)
            );
        }

        inline void Remove(const u08 piece, const u08 color, const u08 sq)
        {
            const s00 vIdx =  color      * ColorStride + piece * PieceStride +  sq      ;
            const s00 uIdx = (color ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            Accumulator& accumulator = Accumulators[AccumulatorP];

            ArraySub(
                accumulator[0],
                Slice<HiddenSize>(FeatureWeight, vIdx * HiddenSize)
            );
            ArraySub(
                accumulator[1],
                Slice<HiddenSize>(FeatureWeight, uIdx * HiddenSize)
            );
        }

        inline O Evaluate(const u08 perspective)
        {
            assert(perspective < 2);

            const Accumulator& accumulator = Accumulators[AccumulatorP];

            O output = ActivateFlattenAndForward<ActivationFunction, I, O>(
                accumulator[perspective    ],
                accumulator[perspective ^ 1],
                OutputWeight,
                OutputBias
            )[0];

            return output * Scale / (QuantizationFeature * QuantizationOutput);
        }

    };


} // MantaRay

#endif //MANTARAY_PERSPECTIVE_H
