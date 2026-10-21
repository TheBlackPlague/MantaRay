//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_ACCUMULATORSTACK_H
#define MANTARAY_RUNTIME_ACCUMULATORSTACK_H

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <memory>
#include <span>
#include <type_traits>

#include "Delta.h"
#include "FinnyTable.h"
#include "Network.h"
#include "../Architecture/FeatureSet.h"
#include "../Backend/Target.h"
#include "../Backend/Kernel/Update.h"
#include "../Common/Alignment.h"
#include "../Common/Integral.h"

namespace MantaRay
{

    enum class UpdatePolicy : u08 { Eager, Lazy };

    template<typename A, s00 Capacity, UpdatePolicy Policy = UpdatePolicy::Lazy, typename ISA = Backend::Native>
    class alignas(Alignment) AccumulatorStack
    {

        using N = Network<A, ISA>;
        using T = N::Accumulator;
        using F = A::Transformer::Features;

        constexpr static s00  H        = N::HiddenSize;
        constexpr static bool Lazy     = Policy == UpdatePolicy::Lazy;
        constexpr static bool Bucketed = F::KeyCount > 1;
        constexpr static bool Counted  = N::Buckets::Count > 1;
        constexpr static u08  Computed = 0b11;

        static_assert(Capacity > 0, "the stack needs room for the root");

        using KeyArray   = std::conditional_t<Bucketed, std::array<std::array<u08, 2>, Capacity>, Absent<0>>;
        using CountArray = std::conditional_t<Counted , std::array<u08, Capacity>                , Absent<1>>;
        using Table      = std::conditional_t<Bucketed, FinnyTable<A, ISA>                       , Absent<2>>;
        using Stamp      = std::conditional_t<Bucketed, u64                                      , Absent<3>>;

        alignas(Alignment) std::array<std::array<std::array<T, H>, 2>, Capacity> Accumulators {};

        std::array<Delta, Capacity> Deltas {};
        std::array<u08  , Capacity> Flags  {};

        NO_UNIQUE_ADDRESS KeyArray   Keys            {};
        NO_UNIQUE_ADDRESS CountArray PieceCounts     {};
        NO_UNIQUE_ADDRESS Table      Cache           {};
        NO_UNIQUE_ADDRESS Stamp      CacheGeneration {};

        const N* Model;

        s00 Top = 0;

        [[clang::always_inline]]
              T* At(const s00 ply, const u08 perspective)       noexcept
        { return std::assume_aligned<Alignment>(Accumulators[ply][perspective].data()); }
        [[clang::always_inline]]
        const T* At(const s00 ply, const u08 perspective) const noexcept
        { return std::assume_aligned<Alignment>(Accumulators[ply][perspective].data()); }

        [[clang::always_inline]]
        u08 Key(const s00 ply, const u08 perspective) const noexcept
        {
            if constexpr (Bucketed) return Keys[ply][perspective];
            else                    return 0;
        }

        [[clang::always_inline]]
        const T* Row(const u08 perspective, const u08 key, const u16 code) const noexcept
        { return Model->FeatureRow(F::Index(perspective, key, code)); }

        template<s00 Subs, s00 Adds, u08 Perspectives>
        [[clang::always_inline]]
        void Step(const s00 ply, const Delta& delta) noexcept
        {
            constexpr auto Selected = [](const u08 perspective) { return (Perspectives >> perspective & 1) != 0; };

            if constexpr (Subs + Adds == 0) {
                #pragma clang loop unroll(full)
                for (u08 perspective = 0; perspective < 2; perspective++)
                    if (Selected(perspective))
                        Backend::Kernel::Copy<ISA, H>(At(ply - 1, perspective), At(ply, perspective));
            } else {
                std::array<std::array<const T*, Subs>, 2> subtract;
                std::array<std::array<const T*, Adds>, 2> add     ;

                #pragma clang loop unroll(full)
                for (u08 perspective = 0; perspective < 2; perspective++) {
                    if (!Selected(perspective)) continue;

                    const u08 key = Key(ply, perspective);

                    for (s00 i = 0; i < Subs; i++) subtract[perspective][i] = Row(perspective, key, delta.Removed(i));
                    for (s00 i = 0; i < Adds; i++) add     [perspective][i] = Row(perspective, key, delta.Inserted  (i));
                }

                #pragma clang loop unroll(full)
                for (u08 perspective = 0; perspective < 2; perspective++)
                    if (Selected(perspective))
                        Backend::Kernel::Update<ISA, H, Subs, Adds>(
                            At(ply - 1, perspective),
                            At(ply    , perspective),
                            subtract[perspective],
                            add     [perspective]
                        );
            }
        }

        template<u08 Perspectives>
        [[clang::noinline]]
        void AdvanceRare(const s00 ply, const Delta& delta) noexcept
        {
            switch (delta.Shape()) {
                case Delta::ShapeOf(0, 0): Step<0, 0, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(0, 1): Step<0, 1, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(0, 2): Step<0, 2, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(1, 0): Step<1, 0, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(1, 2): Step<1, 2, Perspectives>(ply, delta); break;
                default                  : Step<2, 0, Perspectives>(ply, delta); break;
            }
        }

        template<u08 Perspectives = Computed>
        [[clang::always_inline]]
        void Advance(const s00 ply, const Delta& delta) noexcept
        {
            switch (delta.Shape()) {
                case Delta::ShapeOf(1, 1):  Step<1, 1, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(2, 1):  Step<2, 1, Perspectives>(ply, delta); break;
                case Delta::ShapeOf(2, 2):  Step<2, 2, Perspectives>(ply, delta); break;
                default                  : AdvanceRare<Perspectives>(ply, delta); break;
            }
        }

        [[clang::always_inline]]
        void Advance(const s00 ply, const Delta& delta, const u08 perspectives) noexcept
        {
            switch (perspectives) {
                case 0b01: Advance<0b01>(ply, delta); break;
                case 0b10: Advance<0b10>(ply, delta); break;
                default  : Advance<0b11>(ply, delta); break;
            }
        }

        [[clang::always_inline]]
        void Track(const s00 ply, const Delta& delta) noexcept
        {
            if constexpr (Counted)
                PieceCounts[ply] = static_cast<u08>(PieceCounts[ply - 1] + delta.InsertedCount() - delta.RemovedCount());

            if constexpr (Bucketed) {
                Keys[ply] = Keys[ply - 1];

                #pragma clang loop unroll(full)
                for (s00 i = 0; i < 2; i++) {
                    if (i == delta.InsertedCount()) break;

                    const u16 code  = delta.Inserted(i);
                    const u08 color = ColorOf(code);

                    if (PieceOf(code) == KingPiece) Keys[ply][color] = F::Key(color, SquareOf(code));
                }
            }
        }

        [[clang::always_inline]]
        bool Ready(const s00 ply) const noexcept
        {
            if constexpr (Bucketed) return Flags[ply - 1] == Computed && Keys[ply] == Keys[ply - 1];
            else                    return Flags[ply - 1] == Computed;
        }

        [[clang::always_inline]]
        u08 Unchanged(const s00 ply) const noexcept requires (Bucketed)
        {
            return (Keys[ply][0] == Keys[ply - 1][0]) | (Keys[ply][1] == Keys[ply - 1][1]) << 1;
        }

        [[clang::noinline]]
        void CatchUp() noexcept requires (!Bucketed)
        {
            const s00 top = Top;

            s00 ply = top - 1;

            while (Flags[ply - 1] != Computed) ply--;

            for (; ply <= top; ply++) {
                Advance(ply, Deltas[ply]);

                Flags[ply] = Computed;
            }
        }

        template<typename PieceBoards>
        [[clang::noinline]]
        void CatchUp(const PieceBoards& boards) noexcept requires (Bucketed)
        {
            const s00 top = Top;

            std::array<s00, 2> base;

            for (u08 perspective = 0; perspective < 2; perspective++) {
                s00 ply = top;

                while (!(Flags[ply] >> perspective & 1) && Keys[ply][perspective] == Keys[ply - 1][perspective]) ply--;

                if (Flags[ply] >> perspective & 1) base[perspective] = ply;
                else {
                    Cache.Refresh(*Model, perspective, Keys[top][perspective], boards, At(top, perspective));

                    Flags[top] |= 1 << perspective;
                    base [perspective] = top;
                }
            }

            for (s00 ply = std::min(base[0], base[1]) + 1; ply <= top; ply++) {
                const u08 perspectives = (base[0] < ply) | (base[1] < ply) << 1;

                Advance(ply, Deltas[ply], perspectives);

                Flags[ply] |= perspectives;
            }
        }

        template<typename PieceBoards>
        [[clang::noinline]]
        void RefreshTop(const PieceBoards& boards) noexcept requires (Bucketed)
        {
            const s00 top = Top;

            for (u08 perspective = 0; perspective < 2; perspective++)
                if (!(Flags[top] >> perspective & 1))
                    Cache.Refresh(*Model, perspective, Keys[top][perspective], boards, At(top, perspective));

            Flags[top] = Computed;
        }

        [[clang::noinline]]
        void AdvanceOne(const s00 ply, const Delta& delta, const u08 perspectives) noexcept requires (Bucketed)
        {
            if (perspectives == 0b01) Advance<0b01>(ply, delta);
            else                      Advance<0b10>(ply, delta);
        }

        [[clang::always_inline]]
        s00 Bucket() const noexcept
        {
            if constexpr (Counted) return N::Bucket(PieceCounts[Top]);
            else                   return 0;
        }

        [[clang::always_inline]]
        i32 Forward(const u08 stm) const noexcept
        {
            assert(stm < 2);

            return Model->Forward(At(Top, stm), At(Top, stm ^ 1), Bucket());
        }

        template<s00 Subs, s00 Adds>
        [[clang::always_inline]]
        i32 Fuse(const s00 ply, const Delta& delta, const u08 stm) noexcept
        {
            assert(ply == Top && stm < 2);

            Backend::Kernel::Transition<T, Subs, Adds> transition;

            #pragma clang loop unroll(full)
            for (u08 perspective = 0; perspective < 2; perspective++) {
                const u08 key = Key(ply, perspective);

                transition.Sources     [perspective] = At(ply - 1, perspective);
                transition.Destinations[perspective] = At(ply    , perspective);

                for (s00 i = 0; i < Subs; i++)
                    transition.Subtract[perspective][i] = Row(perspective, key, delta.Removed(i));
                for (s00 i = 0; i < Adds; i++)
                    transition.Add     [perspective][i] = Row(perspective, key, delta.Inserted  (i));
            }

            transition.SideToMove = stm;

            const s00 bucket = Bucket();

            Flags[ply] = Computed;

            return Model->ForwardFused(transition, bucket);
        }

        template<typename Settle>
        [[clang::always_inline]]
        i32 EvaluateFused(const u08 stm, const Settle& settle) noexcept
        {
            const s00 top = Top;

            if (Flags[top] != Computed) {
                if (!Ready(top)) settle();
                else {
                    const Delta& delta = Deltas[top];

                    switch (delta.Shape()) {
                        case Delta::ShapeOf(1, 1):     return Fuse<1, 1>(top, delta, stm);
                        case Delta::ShapeOf(2, 1):     return Fuse<2, 1>(top, delta, stm);
                        case Delta::ShapeOf(2, 2):     return Fuse<2, 2>(top, delta, stm);
                        default                  : AdvanceRare<Computed>(top, delta     ); break;
                    }

                    Flags[top] = Computed;
                }
            }

            return Forward(stm);
        }

        public:
        using Accumulator = T;

        constexpr static bool Fused = Lazy && N::Fusable;

        explicit AccumulatorStack(const N& network) noexcept : Model(&network)
        {
            ClearCache();

            for (u08 perspective = 0; perspective < 2; perspective++)
                Backend::Kernel::Copy<ISA, H>(Model->FeatureBias(), At(0, perspective));

            Flags[0] = Computed;
        }

        AccumulatorStack(const N&&) = delete;

        template<typename PieceBoards>
        void Reset(const PieceBoards& boards) noexcept
        {
            if constexpr (Bucketed) {
                if (CacheGeneration != Model->Generation()) ClearCache();

                for (u08 perspective = 0; perspective < 2; perspective++) {
                    const u64 king = boards(KingPiece, perspective);

                    assert(king != 0);

                    const u08 key = F::Key(perspective, static_cast<u08>(std::countr_zero(king) & 63));

                    Cache.Refresh(*Model, perspective, key, boards, At(0, perspective));

                    Keys[0][perspective] = key;
                }
            } else {
                std::array<u16, F::MaxActive> codes;

                s00 count = 0;

                for (u08 color = 0; color < 2; color++)
                for (u08 piece = 0; piece < 6; piece++)
                for (u64 board = boards(piece, color); board != 0; board &= board - 1) {
                    assert(count < F::MaxActive);

                    if (count < F::MaxActive)
                        codes[count++] = PieceCode(piece, color, static_cast<u08>(std::countr_zero(board)));
                }

                for (u08 perspective = 0; perspective < 2; perspective++) {
                    std::array<const T*, F::MaxActive> rows;

                    for (s00 i = 0; i < count; i++) rows[i] = Row(perspective, 0, codes[i]);

                    Backend::Kernel::Refresh<ISA, H>(
                        Model->FeatureBias(),
                        At(0, perspective),
                        nullptr,
                        {},
                        std::span(rows).first(count)
                    );
                }
            }

            if constexpr (Counted) {
                s00 pieces = 0;

                for (u08 color = 0; color < 2; color++)
                for (u08 piece = 0; piece < 6; piece++)
                    pieces += std::popcount(boards(piece, color));

                PieceCounts[0] = static_cast<u08>(std::min<s00>(pieces, 255));
            }

            Top      = 0;
            Flags[0] = Computed;
        }

        void Push(const Delta& delta) noexcept
        {
            assert(Top < Capacity - 1);

            const s00 top = ++Top;

            Track(top, delta);

            if constexpr (Lazy) {
                Deltas[top] = delta;
                Flags [top] = 0;
            } else if constexpr (Bucketed) {
                const u08 ready = Flags[top - 1] & Unchanged(top);

                Flags[top] = ready;

                if      (ready == Computed) Advance   (top, delta       );
                else if (ready != 0       ) AdvanceOne(top, delta, ready);
            } else Advance(top, delta);
        }

        [[clang::always_inline]]
        void Pop() noexcept
        {
            assert(Top > 0);

            Top--;
        }

        [[clang::always_inline]]
        void Materialize() noexcept requires (!Bucketed)
        {
            if constexpr (Lazy) {
                const s00 top = Top;

                if (Flags[top] == Computed) return;

                if (Ready(top)) {
                    Advance(top, Deltas[top]);

                    Flags[top] = Computed;
                } else CatchUp();
            }
        }

        template<typename PieceBoards>
        [[clang::always_inline]]
        void Materialize(const PieceBoards& boards) noexcept
        {
            if constexpr (!Bucketed) Materialize();
            else {
                const s00 top = Top;

                if (Flags[top] == Computed) return;

                if constexpr (Lazy) {
                    if (Ready(top)) {
                        Advance(top, Deltas[top]);

                        Flags[top] = Computed;
                    } else CatchUp(boards);
                } else RefreshTop(boards);
            }
        }

        [[nodiscard]]
        i32 Evaluate(const u08 stm) noexcept requires (!Bucketed)
        {
            if constexpr (Fused) return EvaluateFused(stm, [this] { CatchUp(); });
            else {
                Materialize();

                return Forward(stm);
            }
        }

        template<typename PieceBoards>
        [[nodiscard]]
        i32 Evaluate(const u08 stm, const PieceBoards& boards) noexcept
        {
            if constexpr (!Bucketed) return Evaluate(stm);
            else if constexpr (Fused) return EvaluateFused(stm, [&] { CatchUp(boards); });
            else {
                Materialize(boards);

                return Forward(stm);
            }
        }

        void ClearCache() noexcept
        {
            if constexpr (Bucketed) {
                Cache.Clear(*Model);

                CacheGeneration = Model->Generation();
            }
        }

        [[nodiscard]]
        s00 Depth() const noexcept { return Top; }

        [[nodiscard]]
        const Accumulator* Values(const u08 perspective) const noexcept
        {
            assert(perspective < 2 && (!(Lazy || Bucketed) || Flags[Top] >> perspective & 1));

            return At(Top, perspective);
        }

    };

}

#endif //MANTARAY_RUNTIME_ACCUMULATORSTACK_H
