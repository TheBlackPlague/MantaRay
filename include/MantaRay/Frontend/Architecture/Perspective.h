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

#include "Common/Accumulator.h"

namespace MantaRay
{

    template<QuantizedInteger I,
             QuantizedInteger O,
             auto ActivationFunction,
             s00  InputSize,
             s00 HiddenSize,
             s00 OutputSize,
             I Scale,
             I QuantizationFeature,
             I QuantizationOutput>
    class Perspective
    {

        static_assert(Scale > 0 && QuantizationFeature > 127 && QuantizationOutput > 31,
                      "These scale and quantization constants don't seem right.");
        static_assert(InputSize >= 64 * 6 * 2, "The input layer cannot represent every piece-square feature.");
        static_assert(OutputSize > 0, "The output layer size cannot be zero.");

        constexpr static s00 ColorStride = 64 * 6;
        constexpr static s00 PieceStride = 64    ;

        static_assert(sizeof(NArray<I, InputSize, HiddenSize>) == sizeof(I) * InputSize * HiddenSize,
                      "First-layer weight storage must remain tightly packed.");
        static_assert(sizeof(NArray<I, OutputSize, HiddenSize * 2>) == sizeof(I) * OutputSize * HiddenSize * 2,
                      "Output-layer weight storage must remain tightly packed.");

        ALIGN NArray<I,  InputSize,     HiddenSize> L0Weight;
        ALIGN Array <I,                 HiddenSize> L0Bias  ;
        ALIGN NArray<I, OutputSize, HiddenSize * 2> L1Weight;
        ALIGN Array <I,                 OutputSize> L1Bias  ;

        public:
        Perspective() = default;

        Perspective(BinaryFileStream<>& stream)
        {
            stream.ReadArray(L0Weight);
            stream.ReadArray(L0Bias  );
            stream.ReadArray(L1Weight);
            stream.ReadArray(L1Bias  );
        }

        Perspective(BinaryMemoryStream& stream)
        {
            stream.ReadArray(L0Weight);
            stream.ReadArray(L0Bias  );
            stream.ReadArray(L1Weight);
            stream.ReadArray(L1Bias  );
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
            ss << " | " << "Scale                : " <<                       Scale << std::endl;
            ss << " | " << "QuantizationFeature  : " <<         QuantizationFeature << std::endl;
            ss << " | " << "QuantizationOutput   : " <<          QuantizationOutput << std::endl;

            return ss.str();
        }

        void Refresh(Accumulator<I, HiddenSize>& accumulator) const { accumulator.Bias(L0Bias); }

        [[clang::always_inline]]
        void Move(const u08 piece, const u08 color, const u08 from, const u08 to,
                  Accumulator<I, HiddenSize>& accumulator) const
        {
            assert(piece < 6 && color < 2 && from < 64 && to < 64);

            const s00 fromIdxV =  color      * ColorStride + piece * PieceStride +  from      ;
            const s00 fromIdxU = (color ^ 1) * ColorStride + piece * PieceStride + (from ^ 56);
            const s00   toIdxV =  color      * ColorStride + piece * PieceStride +  to        ;
            const s00   toIdxU = (color ^ 1) * ColorStride + piece * PieceStride + (to   ^ 56);

            ArraySubAdd(
                accumulator[0],
                L0Weight[fromIdxV],
                L0Weight[  toIdxV]
            );
            ArraySubAdd(
                accumulator[1],
                L0Weight[fromIdxU],
                L0Weight[  toIdxU]
            );
        }

        [[clang::always_inline]]
        void Insert(const u08 piece, const u08 color, const u08 sq, Accumulator<I, HiddenSize>& accumulator) const
        {
            assert(piece < 6 && color < 2 && sq < 64);

            const s00 vIdx =  color      * ColorStride + piece * PieceStride +  sq      ;
            const s00 uIdx = (color ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            ArrayAdd(
                accumulator[0],
                L0Weight[vIdx]
            );
            ArrayAdd(
                accumulator[1],
                L0Weight[uIdx]
            );
        }

        [[clang::always_inline]]
        void Remove(const u08 piece, const u08 color, const u08 sq, Accumulator<I, HiddenSize>& accumulator) const
        {
            assert(piece < 6 && color < 2 && sq < 64);

            const s00 vIdx =  color      * ColorStride + piece * PieceStride +  sq      ;
            const s00 uIdx = (color ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            ArraySub(
                accumulator[0],
                L0Weight[vIdx]
            );
            ArraySub(
                accumulator[1],
                L0Weight[uIdx]
            );
        }

        [[clang::always_inline]]
        O Evaluate(const u08 perspective, const Accumulator<I, HiddenSize>& accumulator) const
        {
            assert(perspective < 2);

            O output = ActivateFlattenAndForward<ActivationFunction, I, O>(
                accumulator[perspective    ],
                accumulator[perspective ^ 1],
                L1Weight,
                L1Bias
            )[0];

            return WrapMul(output, static_cast<O>(Scale)) / (QuantizationFeature * QuantizationOutput);
        }

    };


} // MantaRay

#endif //MANTARAY_PERSPECTIVE_H
