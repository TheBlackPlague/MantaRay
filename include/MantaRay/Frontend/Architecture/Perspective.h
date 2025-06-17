//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_PERSPECTIVE_H
#define MANTARAY_PERSPECTIVE_H

#include <sstream>
#include <string>

#include "../../Backend/Kernel/ActivateFlattenAndForward.h"
#include "../../Backend/Kernel/ArrayOperate.h"

#include "../IO/BinaryFileStream.h"
#include "../IO/BinaryMemoryStream.h"

#include "Common/Accumulator.h"

namespace MantaRay
{

    template<QuantizedInteger I,
             QuantizedInteger O,
             typename ActivationFunction,
             usize  InputSize,
             usize HiddenSize,
             usize OutputSize,
             I Scale,
             I QuantizationFeature,
             I QuantizationOutput>
    class Perspective
    {

        static_assert(Scale > 0 && QuantizationFeature > 127 && QuantizationOutput > 31,
                      "These scale and quantization constants don't seem right.");

        constexpr static usize ColorStride = 64 * 6;
        constexpr static usize PieceStride = 64    ;

        HWY_ALIGN Array<I,  InputSize *     HiddenSize> L0Weight;
        HWY_ALIGN Array<I,                  HiddenSize> L0Bias  ;
        HWY_ALIGN Array<I, HiddenSize * 2 * OutputSize> L1Weight;
        HWY_ALIGN Array<I,                  OutputSize> L1Bias  ;

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
            ss << " | " << "QuantizationOutput   : " <<         QuantizationOutput  << std::endl;

            return ss.str();
        }

        void Refresh(Accumulator<I, HiddenSize>& accumulator) const { accumulator.Bias(L0Bias); }

        struct AccumulatorUpdate {};

        struct AccumulatorUpdateNormal : AccumulatorUpdate
        {

            u08  Piece;
            u08   Side;
            u08 Origin;
            u08 Target;

        };

        [[clang::always_inline]]
        void Normal(const u08 piece, const u08 side, const u08 origin, const u08 target,
                    Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize originIdx0 =  side      * ColorStride + piece * PieceStride +  origin      ;
            const usize originIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (origin ^ 56);
            const usize targetIdx0 =  side      * ColorStride + piece * PieceStride +  target      ;
            const usize targetIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (target ^ 56);

            ArrayOperate<Sub, Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, originIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx0 * HiddenSize)
            );
            ArrayOperate<Sub, Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, originIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdateCapture : AccumulatorUpdate
        {

            u08 VictimPiece;

            u08  Piece;
            u08   Side;
            u08 Origin;
            u08 Target;

        };

        [[clang::always_inline]]
        void Capture(const u08 victimPiece, const u08 piece, const u08 side, const u08 origin, const u08 target,
                     Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize victimIdx0 = (side ^ 1) * ColorStride + victimPiece * PieceStride +  target      ;
            const usize victimIdx1 =  side      * ColorStride + victimPiece * PieceStride + (target ^ 56);

            const usize originIdx0 =  side      * ColorStride + piece * PieceStride +  origin      ;
            const usize originIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (origin ^ 56);
            const usize targetIdx0 =  side      * ColorStride + piece * PieceStride +  target      ;
            const usize targetIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (target ^ 56);

            ArrayOperate<Sub, Sub, Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, victimIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx0 * HiddenSize)
            );

            ArrayOperate<Sub, Sub, Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, victimIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdatePromotion : AccumulatorUpdate
        {

            u08 PromotionPiece;

            u08  Piece;
            u08   Side;
            u08 Origin;
            u08 Target;

        };

        [[clang::always_inline]]
        void Promotion(const u08 promotionPiece, const u08 piece, const u08 side, const u08 origin, const u08 target,
                       Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize originIdx0 =  side      * ColorStride +          piece * PieceStride +  origin      ;
            const usize originIdx1 = (side ^ 1) * ColorStride +          piece * PieceStride + (origin ^ 56);
            const usize targetIdx0 =  side      * ColorStride + promotionPiece * PieceStride +  target      ;
            const usize targetIdx1 = (side ^ 1) * ColorStride + promotionPiece * PieceStride + (target ^ 56);

            ArrayOperate<Sub, Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, originIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx0 * HiddenSize)
            );
            ArrayOperate<Sub, Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, originIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdatePromotionCapture : AccumulatorUpdate
        {

            u08 PromotionPiece;

            u08 VictimPiece;

            u08  Piece;
            u08   Side;
            u08 Origin;
            u08 Target;

        };

        [[clang::always_inline]]
        void PromotionCapture(const u08 promotionPiece, const u08 victimPiece, const u08 piece, const u08 side,
                              const u08 origin, const u08 target,
                              Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize victimIdx0 = (side ^ 1) * ColorStride + victimPiece * PieceStride +  target      ;
            const usize victimIdx1 =  side      * ColorStride + victimPiece * PieceStride + (target ^ 56);

            const usize originIdx0 =  side      * ColorStride +          piece * PieceStride +  origin      ;
            const usize originIdx1 = (side ^ 1) * ColorStride +          piece * PieceStride + (origin ^ 56);
            const usize targetIdx0 =  side      * ColorStride + promotionPiece * PieceStride +  target      ;
            const usize targetIdx1 = (side ^ 1) * ColorStride + promotionPiece * PieceStride + (target ^ 56);

            ArrayOperate<Sub, Sub, Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, victimIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx0 * HiddenSize)
            );
            ArrayOperate<Sub, Sub, Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, victimIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdateCastle : AccumulatorUpdate
        {

            u08 Side;

            u08 OriginKing;
            u08 TargetKing;
            u08 OriginRook;
            u08 TargetRook;

        };

        [[clang::always_inline]]
        void Castle(const u08 side, const u08 originK, const u08 targetK, const u08 originR, const u08 targetR,
                    Accumulator<I, HiddenSize>& accumulator) const
        {
            constexpr static u08 King = 5;
            constexpr static u08 Rook = 3;

            const usize originKIdx0 =  side      * ColorStride + King * PieceStride +  originK      ;
            const usize originKIdx1 = (side ^ 1) * ColorStride + King * PieceStride + (originK ^ 56);
            const usize targetKIdx0 =  side      * ColorStride + King * PieceStride +  targetK      ;
            const usize targetKIdx1 = (side ^ 1) * ColorStride + King * PieceStride + (targetK ^ 56);
            const usize originRIdx0 =  side      * ColorStride + Rook * PieceStride +  originR      ;
            const usize originRIdx1 = (side ^ 1) * ColorStride + Rook * PieceStride + (originR ^ 56);
            const usize targetRIdx0 =  side      * ColorStride + Rook * PieceStride +  targetR      ;
            const usize targetRIdx1 = (side ^ 1) * ColorStride + Rook * PieceStride + (targetR ^ 56);

            ArrayOperate<Sub, Sub, Add, Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, originKIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originRIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetKIdx0 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetRIdx0 * HiddenSize)
            );
            ArrayOperate<Sub, Sub, Add, Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, originKIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, originRIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetKIdx1 * HiddenSize),
                Slice<HiddenSize>(L0Weight, targetRIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdateInsert : AccumulatorUpdate
        {

            u08 Piece;
            u08  Side;
            u08    Sq;

        };

        [[clang::always_inline]]
        void Insert(const u08 piece, const u08 side, const u08 sq, Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize sqIdx0 =  side      * ColorStride + piece * PieceStride +  sq      ;
            const usize sqIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            ArrayOperate<Add>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, sqIdx0 * HiddenSize)
            );
            ArrayOperate<Add>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, sqIdx1 * HiddenSize)
            );
        }

        struct AccumulatorUpdateRemove : AccumulatorUpdateInsert {};

        [[clang::always_inline]]
        void Remove(const u08 piece, const u08 side, const u08 sq, Accumulator<I, HiddenSize>& accumulator) const
        {
            const usize sqIdx0 =  side      * ColorStride + piece * PieceStride +  sq      ;
            const usize sqIdx1 = (side ^ 1) * ColorStride + piece * PieceStride + (sq ^ 56);

            ArrayOperate<Sub>(
                accumulator[0],
                Slice<HiddenSize>(L0Weight, sqIdx0 * HiddenSize)
            );
            ArrayOperate<Sub>(
                accumulator[1],
                Slice<HiddenSize>(L0Weight, sqIdx1 * HiddenSize)
            );
        }

        template<typename AccumulatorUpdateT>
        [[clang::always_inline]]
        void DispatchUpdate(const AccumulatorUpdateT& update, Accumulator<I, HiddenSize>& accumulator) const
        requires std::is_base_of_v<AccumulatorUpdate, AccumulatorUpdateT>
        {
            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdateNormal>) {
                const auto& normalUpdate = static_cast<const AccumulatorUpdateNormal&>(update);
                Normal(
                    normalUpdate.Piece,
                    normalUpdate.Side,
                    normalUpdate.Origin,
                    normalUpdate.Target,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdateCapture>) {
                const auto& captureUpdate = static_cast<const AccumulatorUpdateCapture&>(update);
                Capture(
                    captureUpdate.VictimPiece,
                    captureUpdate.Piece,
                    captureUpdate.Side,
                    captureUpdate.Origin,
                    captureUpdate.Target,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdatePromotion>) {
                const auto& promotionUpdate = static_cast<const AccumulatorUpdatePromotion&>(update);
                Promotion(
                    promotionUpdate.PromotionPiece,
                    promotionUpdate.Piece,
                    promotionUpdate.Side,
                    promotionUpdate.Origin,
                    promotionUpdate.Target,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdatePromotionCapture>) {
                const auto& promotionCaptureUpdate = static_cast<const AccumulatorUpdatePromotionCapture&>(update);
                PromotionCapture(
                    promotionCaptureUpdate.PromotionPiece,
                    promotionCaptureUpdate.VictimPiece,
                    promotionCaptureUpdate.Piece,
                    promotionCaptureUpdate.Side,
                    promotionCaptureUpdate.Origin,
                    promotionCaptureUpdate.Target,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdateCastle>) {
                const auto& castleUpdate = static_cast<const AccumulatorUpdateCastle&>(update);
                Castle(
                    castleUpdate.Side,
                    castleUpdate.OriginKing,
                    castleUpdate.TargetKing,
                    castleUpdate.OriginRook,
                    castleUpdate.TargetRook,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdateInsert>) {
                const auto& insertUpdate = static_cast<const AccumulatorUpdateInsert&>(update);
                Insert(
                    insertUpdate.Piece,
                    insertUpdate.Side,
                    insertUpdate.Sq,
                    accumulator
                );
            }

            if constexpr (std::is_same_v<AccumulatorUpdateT, AccumulatorUpdateRemove>) {
                const auto& removeUpdate = static_cast<const AccumulatorUpdateRemove&>(update);
                Remove(
                    removeUpdate.Piece,
                    removeUpdate.Side,
                    removeUpdate.Sq,
                    accumulator
                );
            }
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

            return output * Scale / (QuantizationFeature * QuantizationOutput);
        }

    };


} // MantaRay

#endif //MANTARAY_PERSPECTIVE_H
