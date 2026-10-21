//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_UPDATE_H
#define MANTARAY_BACKEND_KERNEL_UPDATE_H

#include <algorithm>
#include <array>
#include <span>
#include <type_traits>

#include "../../Common.h"
#include "../../Architecture/Signal.h"

namespace MantaRay::Backend::Kernel
{

    namespace Detail
    {

        template<typename ISA, s00 H, typename T>
        consteval s00 Lanes()
        {
            static_assert(H * sizeof(T) % VectorBytes == 0, "kernel width must be a multiple of 64 bytes");
            static_assert(VectorBytes   % ISA::Bytes  == 0, "ISA vector must divide 64 bytes"            );

            return ISA::Bytes / sizeof(T);
        }

        consteval s00 Tile(const s00 vectors, const s00 limit)
        {
            for (s00 tile = std::min(vectors, limit); tile > 1; tile--) if (vectors % tile == 0) return tile;

            return 1;
        }

        template<typename ISA, typename T>
        [[clang::always_inline]]
        ISA::Vector Add(const typename ISA::Vector left, const typename ISA::Vector right)
        {
            static_assert(std::is_same_v<T, i16> || std::is_same_v<T, i32>, "accumulators are i16 or i32");

            if (std::is_same_v<T, i16>) return ISA::AddI16(left, right);

            return ISA::AddI32(left, right);
        }

        template<typename ISA, typename T>
        [[clang::always_inline]]
        ISA::Vector Sub(const typename ISA::Vector left, const typename ISA::Vector right)
        {
            static_assert(std::is_same_v<T, i16> || std::is_same_v<T, i32>, "accumulators are i16 or i32");

            if (std::is_same_v<T, i16>) return ISA::SubI16(left, right);

            return ISA::SubI32(left, right);
        }

        template<typename ISA, typename T, s00 H, s00 Subs, s00 Adds>
        [[clang::always_inline]]
        ISA::Vector Apply(
                  typename ISA::Vector                       value   ,
            const std::array<const std::array<T, H>*, Subs>& subtract,
            const std::array<const std::array<T, H>*, Adds>& add     ,
            const s00                                        offset
        )
        {
            const auto row = [&](const Array<T, H>* pointer) { return ISA::Load(*pointer, offset); };

            if (Subs + Adds <= 2) {
                for (const Array<T, H>* pointer : subtract) value = Sub<ISA, T>(value, row(pointer));
                for (const Array<T, H>* pointer : add     ) value = Add<ISA, T>(value, row(pointer));

                return value;
            }

            if constexpr (Adds == 1) {
                const auto left  = Add<ISA, T>(value           , row(add     [0]));
                const auto right = Add<ISA, T>(row(subtract[0]), row(subtract[1]));

                return Sub<ISA, T>(left, right);
            } else if constexpr (Subs == 1) {
                const auto left  = Sub<ISA, T>(value      , row(subtract[0]));
                const auto right = Add<ISA, T>(row(add[0]), row(add     [1]));

                return Add<ISA, T>(left, right);
            } else {
                const auto left  = Add<ISA, T>(Sub<ISA, T>(value, row(subtract[0])), row(add     [0]));
                const auto right = Sub<ISA, T>(row(add[1])                         , row(subtract[1]));

                return Add<ISA, T>(left, right);
            }
        }

        template<typename ISA, s00 H, typename T, typename... D>
        [[clang::always_inline]]
        void Refresh(
            const std::array<T, H>&                        base        ,
            const std::span<const std::array<T, H>* const> subtract    ,
            const std::span<const std::array<T, H>* const> add         ,
                  D&...                                    destinations
        )
        {
            constexpr s00 Lanes = Detail::Lanes<ISA, H, T>();
            constexpr s00 Tile  = Detail::Tile(H / Lanes, ISA::Registers - 4);
            constexpr s00 Width = Tile * Lanes;

            static_assert(H % Width == 0, "refresh tiles must cover the accumulator exactly");

            for (s00 offset = 0; offset < H; offset += Width) {
                Array<typename ISA::Vector, Tile> values;

                UNROLL(full)
                for (s00 i = 0; i < Tile; i++) values[i] = ISA::Load(base, offset + i * Lanes);

                for (const Array<T, H>* row : subtract) {
                    UNROLL(full)
                    for (s00 i = 0; i < Tile; i++)
                        values[i] = Sub<ISA, T>(values[i], ISA::Load(*row, offset + i * Lanes));
                }

                for (const Array<T, H>* row : add) {
                    UNROLL(full)
                    for (s00 i = 0; i < Tile; i++)
                        values[i] = Add<ISA, T>(values[i], ISA::Load(*row, offset + i * Lanes));
                }

                ([&] {
                    UNROLL(full)
                    for (s00 i = 0; i < Tile; i++) ISA::Store(destinations, offset + i * Lanes, values[i]);
                }(), ...);
            }
        }

    }

    template<typename T, s00 H, s00 Subs, s00 Adds>
    struct Transition
    {

        const Array<T, 2, H>& Sources     ;
              Array<T, 2, H>& Destinations;

        Array<const Array<T, H>*, 2, Subs> Subtract;
        Array<const Array<T, H>*, 2, Adds> Add     ;

        u08 SideToMove;

    };

    template<typename ISA, typename T, s00 H, s00 Subs, s00 Adds>
    [[clang::always_inline]]
    ISA::Vector Advance(const Transition<T, H, Subs, Adds>& transition, const s00 perspective, const s00 offset)
    {
        static_assert(Subs + Adds > 0       , "a transition without rows is a Copy"       );
        static_assert(Subs <= 2 && Adds <= 2, "a transition has at most two rows per side");

        const auto value = Detail::Apply<ISA, T>(
            ISA::Load(transition.Sources[perspective], offset),
            transition.Subtract[perspective],
            transition.Add     [perspective],
            offset
        );

        ISA::Store(transition.Destinations[perspective], offset, value);

        return value;
    }

    template<typename ISA, s00 H, s00 Subs, s00 Adds, typename T>
    [[clang::always_inline]]
    void Update(
        const std::array<T, H>&                          source     ,
              std::array<T, H>&                          destination,
        const std::array<const std::array<T, H>*, Subs>& subtract   ,
        const std::array<const std::array<T, H>*, Adds>& add
    )
    {
        static_assert(Subs + Adds > 0       , "an update without rows is a Copy"       );
        static_assert(Subs <= 2 && Adds <= 2, "an update has at most two rows per side");

        constexpr s00 Lanes = Detail::Lanes<ISA, H, T>();

        const Array<const Array<T, H>*, Subs> subs = subtract;
        const Array<const Array<T, H>*, Adds> adds = add     ;

        UNROLL(full)
        for (s00 i = 0; i < H; i += Lanes)
            ISA::Store(destination, i, Detail::Apply<ISA, T>(ISA::Load(source, i), subs, adds, i));
    }

    template<typename ISA, s00 H, typename T>
    [[clang::always_inline]]
    void Copy(const std::array<T, H>& source, std::array<T, H>& destination)
    {
        constexpr s00 Lanes = Detail::Lanes<ISA, H, T>();

        UNROLL(full)
        for (s00 i = 0; i < H; i += Lanes) ISA::Store(destination, i, ISA::Load(source, i));
    }

    template<typename ISA, s00 H, typename T>
    [[clang::always_inline]]
    void Refresh(
        const std::array<T, H>&                                              base       ,
              std::array<T, H>&                                              destination,
        const std::span<const std::type_identity_t<std::array<T, H>>* const> subtract   ,
        const std::span<const std::type_identity_t<std::array<T, H>>* const> add
    )
    {
        Detail::Refresh<ISA, H, T>(base, subtract, add, destination);
    }

    template<typename ISA, s00 H, typename T>
    [[clang::always_inline]]
    void Refresh(
        const std::array<T, H>&                                              base       ,
              std::array<T, H>&                                              destination,
                              std::type_identity_t<std::array<T, H>>&        mirror     ,
        const std::span<const std::type_identity_t<std::array<T, H>>* const> subtract   ,
        const std::span<const std::type_identity_t<std::array<T, H>>* const> add
    )
    {
        Detail::Refresh<ISA, H, T>(base, subtract, add, destination, mirror);
    }

}

#endif //MANTARAY_BACKEND_KERNEL_UPDATE_H
