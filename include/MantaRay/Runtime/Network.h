//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_NETWORK_H
#define MANTARAY_RUNTIME_NETWORK_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "../Architecture/Architecture.h"
#include "../Architecture/FeatureSet.h"
#include "../Architecture/OutputBuckets.h"
#include "../Backend/Lowering.h"
#include "../Backend/Target.h"
#include "../Backend/Kernel/Update.h"
#include "../Backend/Stage/Format.h"
#include "../Common.h"
#include "../IO/Loader.h"

namespace MantaRay
{

    namespace Detail
    {

        inline std::atomic<u64> NextGeneration { 1 };

        template<typename B>
        consteval auto BucketsOf(List<>)
        {
            if constexpr (std::is_void_v<B>) return std::type_identity<MaterialCount<1>> {};
            else                             return std::type_identity<B               > {};
        }

        template<typename B, typename L, typename... R>
        consteval auto BucketsOf(List<L, R...>)
        {
            if constexpr (!Selecting<L>) return BucketsOf<B>(List<R...> {});
            else {
                static_assert(
                    std::is_void_v<B> || std::is_same_v<B, typename L::Buckets>,
                    "every Select must use the same output buckets"
                );

                return BucketsOf<typename L::Buckets>(List<R...> {});
            }
        }

        template<typename L>
        consteval s00 TensorBytes()
        {
            if constexpr (!Backend::Stage::IsAffine<L>) return 0;
            else return L::InputSize * L::OutputSize * sizeof(typename L::WeightSpec::Type) +
                                       L::OutputSize * sizeof(typename L::BiasSpec  ::Type);
        }

        template<typename A, typename... L>
        consteval s00 PayloadBytes(List<L...>)
        {
            using T = A::Transformer;

            return T::Features::InputSize * T::OutputSize * sizeof(typename T::WeightSpec::Type) +
                                            T::OutputSize * sizeof(typename T::BiasSpec  ::Type) +
                   (TensorBytes<L>() + ... + 0);
        }

        template<typename... L>
        consteval u16 TensorCount(List<L...>)
        {
            return static_cast<u16>(2 + 2 * (s00 { Backend::Stage::IsAffine<L> } + ... + 0));
        }

    }

    template<typename A, typename ISA = Backend::Native>
    class Network
    {

        using Program = Backend::Program<A, ISA>;
        using T       = Program::Accumulator;
        using F       = A::Transformer::Features;
        using W       = A::Transformer::WeightSpec;
        using B       = A::Transformer::BiasSpec;

        constexpr static s00  H         = A::Transformer::OutputSize;
        constexpr static u08  MinPieces =  2;
        constexpr static u08  MaxPieces = 32;
        constexpr static bool Bucketed  = F::KeyCount > 1;

        using Stamp = std::conditional_t<Bucketed, u64, Absent<0>>;

        constexpr static s00 Payload = Detail::PayloadBytes<A>(typename A::Layers {});
        constexpr static u16 Tensors = Detail::TensorCount    (typename A::Layers {});

        static_assert(H * sizeof(T) % Alignment == 0, "feature rows must stay 64-byte aligned");

        alignas(Alignment) Array<T, F::InputSize, H> FeatureWeights {};
        alignas(Alignment) Array<T,               H> FeatureBiases  {};

        Program::Parameters StageParameters {};

        NO_UNIQUE_ADDRESS Stamp LoadGeneration {};

        LoadResult Decode(IO::Reader& reader)
        {
            std::vector<typename W::Type> weights(F::InputSize * H);
            std::vector<typename B::Type> biases (               H);

            if (const LoadResult result = IO::ReadTensor<W>(reader, std::span(weights)); !result) return result;
            if (const LoadResult result = IO::ReadTensor<B>(reader, std::span(biases )); !result) return result;

            for (s00 i = 0; i < F::InputSize; i++)
            for (s00 j = 0; j < H           ; j++) {
                const s00 index = Backend::Stage::BulletIndex<W>(F::InputSize, H, i, j);

                FeatureWeights[i][j] = static_cast<T>(weights[index]);
            }

            for (s00 j = 0; j < H; j++) FeatureBiases[j] = static_cast<T>(biases[j]);

            return Program::Load(reader, StageParameters);
        }

        public:
        using Accumulator = T;
        using Buckets     = decltype(Detail::BucketsOf<void>(typename A::Layers {}))::type;

        constexpr static s00  HiddenSize = H;
        constexpr static bool Fusable    = Program::Fusable;

        [[nodiscard]]
        LoadResult Load(const std::span<const std::byte> bytes)
        {
            if (const LoadResult result = IO::CheckSize(bytes, Payload, Tensors); !result) return result;

            const auto staging = std::make_unique<Network>();

            IO::Reader reader(bytes.first(Payload));

            if (const LoadResult result = staging->Decode(reader); !result) return result;

            assert(reader.Remaining() == 0 && reader.Tensor() == Tensors);

            if constexpr (Bucketed)
                staging->LoadGeneration = Detail::NextGeneration.fetch_add(1, std::memory_order_relaxed);

            *this = *staging;

            return {};
        }

        [[nodiscard]]
        u64 Generation() const noexcept requires (Bucketed) { return LoadGeneration; }

        [[nodiscard]]
        constexpr static std::string_view Describe() { return Program::Describe(); }

        [[nodiscard]]
        i32 Evaluate(const std::span<const u16> pieces, const u08 stm) const
        {
            const auto king = [](const u16 code) { return PieceOf(code) == KingPiece; };

            assert(pieces.size() <= F::MaxActive && stm < 2);
            assert(F::KeyCount == 1 || std::ranges::count_if(pieces, king) == 2);

            Array<u08, 2> kings {};

            for (const u16 code : pieces) if (king(code)) kings[ColorOf(code)] = SquareOf(code);

            const s00 count = std::min<s00>(pieces.size(), F::MaxActive);

            alignas(Alignment) Array<T, 2, H> values;

            for (u08 perspective = 0; perspective < 2; perspective++) {
                const u08 key = F::Key(perspective, kings[perspective]);

                Array<const Array<T, H>*, F::MaxActive> rows;

                for (s00 i = 0; i < count; i++) rows[i] = &FeatureRow(F::Index(perspective, key, pieces[i]));

                Backend::Kernel::Refresh<ISA, H>(
                    FeatureBias(),
                    values[perspective],
                    {},
                    std::span(rows).first(count)
                );
            }

            return Forward(values[stm], values[stm ^ 1], Bucket(static_cast<u08>(count)));
        }

        [[nodiscard]]
        constexpr static s00 Bucket(const u08 pieces) noexcept
        {
            if (Buckets::Count == 1) return 0;

            assert(pieces >= MinPieces);
            assert(pieces <= MaxPieces);

            return Buckets::Of(std::clamp(pieces, MinPieces, MaxPieces));
        }

        [[nodiscard]]
        [[clang::always_inline]]
        const Array<Accumulator, H>& FeatureRow(const u32 index) const noexcept
        {
            assert(index < F::InputSize);

            return FeatureWeights[index];
        }

        [[nodiscard]]
        [[clang::always_inline]]
        const Array<Accumulator, H>& FeatureBias() const noexcept { return FeatureBiases; }

        [[nodiscard]]
        [[clang::always_inline]]
        i32 Forward(const Array<Accumulator, H>& stm, const Array<Accumulator, H>& ntm, const s00 bucket) const noexcept
        {
            return Program::Forward(StageParameters, stm, ntm, bucket);
        }

        template<s00 Subs, s00 Adds>
        [[nodiscard]]
        [[clang::always_inline]]
        i32 ForwardFused(
            const Backend::Kernel::Transition<Accumulator, H, Subs, Adds>& transition,
            const s00                                                      bucket
        ) const noexcept requires (Fusable)
        {
            return Program::template ForwardFused<Subs, Adds>(StageParameters, transition, bucket);
        }

    };

}

#endif //MANTARAY_RUNTIME_NETWORK_H
