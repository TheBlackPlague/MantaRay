//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_DELTA_H
#define MANTARAY_RUNTIME_DELTA_H

#include <cassert>

#include "../Architecture/FeatureSet.h"
#include "../Common/Integral.h"

namespace MantaRay
{

    class Delta
    {

        constexpr static s00 CodeBits = 10;
        constexpr static u64 CodeMask = (u64 { 1 } << CodeBits) - 1;

        constexpr static s00   RemovalShift = 4 * CodeBits;
        constexpr static s00 InsertionShift = RemovalShift + 2;

        constexpr static s00 ShapeShift = 48;
        constexpr static s00 ShapeBits  =  4;

        static_assert(PieceCode(5, 1, 63) <= CodeMask, "every piece code fits a code slot"  );
        static_assert(InsertionShift + 2 <= ShapeShift, "the two counters precede the shape" );
        static_assert(ShapeShift + ShapeBits <= 64   , "the shape (at most 8) fits the word");

        u64 Word = 0;

        [[nodiscard]]
        constexpr u16 Code(const s00 slot) const noexcept
        {
            return static_cast<u16>(Word >> CodeBits * slot & CodeMask);
        }

        public:
        [[nodiscard]]
        constexpr static i32 ShapeOf(const u08 removals, const u08 additions) noexcept
        {
            return removals * 3 + additions;
        }

        constexpr void Remove(const u08 piece, const u08 color, const u08 square) noexcept
        {
            assert(RemovedCount() < 2 && piece < 6 && color < 2 && square < 64);

            Word += (u64 { PieceCode(piece, color, square) } << CodeBits * RemovedCount()) +
                    (u64 { 1 } <<   RemovalShift) + (u64 { ShapeOf(1, 0) } << ShapeShift);
        }

        constexpr void Insert(const u08 piece, const u08 color, const u08 square) noexcept
        {
            assert(InsertedCount() < 2 && piece < 6 && color < 2 && square < 64);

            Word += (u64 { PieceCode(piece, color, square) } << CodeBits * (2 + InsertedCount())) +
                    (u64 { 1 } << InsertionShift) + (u64 { ShapeOf(0, 1) } << ShapeShift);
        }

        constexpr void Move(const u08 piece, const u08 color, const u08 from, const u08 to) noexcept
        {
            Remove(piece, color, from);
            Insert(piece, color, to  );
        }

        [[nodiscard]]
        constexpr u08  RemovedCount() const noexcept { return static_cast<u08>(Word >>   RemovalShift & 3); }
        [[nodiscard]]
        constexpr u08 InsertedCount() const noexcept { return static_cast<u08>(Word >> InsertionShift & 3); }

        [[nodiscard]]
        constexpr i32 Shape() const noexcept
        {
            static_assert(ShapeOf(2, 2) < 1 << ShapeBits, "ShapeBits must hold every shape");

            return static_cast<i32>(Word >> ShapeShift);
        }

        [[nodiscard]]
        constexpr u16  Removed(const s00 index) const noexcept
        {
            assert(index <  RemovedCount());

            return Code(    index);
        }

        [[nodiscard]]
        constexpr u16 Inserted(const s00 index) const noexcept
        {
            assert(index < InsertedCount());

            return Code(2 + index);
        }

    };

}

#endif //MANTARAY_RUNTIME_DELTA_H
