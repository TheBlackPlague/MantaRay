//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_FORMAT_H
#define MANTARAY_IO_FORMAT_H

#include <array>
#include <bit>
#include <concepts>
#include <ios>
#include <limits>
#include <memory>
#include <span>

namespace MantaRay::IO
{

    namespace Detail
    {

        struct FingerprintState
        {

            u64 Value = 14695981039346656037ULL;

            constexpr void Add(const u64 word)
            {
                for (s00 shift = 0; shift != 64; shift += 8)
                    Value = (Value ^ ((word >> shift) & 0xff)) * 1099511628211ULL;
            }

        };

        template<u64 Tag, typename... Children>
        struct FingerprintRecord;

    }

    template<typename T>
    struct Fingerprint;

    namespace Detail
    {

        template<u64 Tag, typename... Children>
        struct FingerprintRecord
        {

            constexpr static void Append(FingerprintState& state)
            {
                state.Add(Tag);
                (Fingerprint<Children>::Append(state), ...);
            }

        };

        template<u64 Tag, typename... Children>
        struct FingerprintList
        {

            constexpr static void Append(FingerprintState& state)
            {
                state.Add(Tag);
                state.Add(sizeof...(Children));
                (Fingerprint<Children>::Append(state), ...);
            }

        };

        template<u64 Tag, i32 L, i32 H>
        struct FingerprintBounds
        {

            constexpr static void Append(FingerprintState& state)
            {
                state.Add(Tag);
                state.Add(static_cast<u64>(L));
                state.Add(static_cast<u64>(H));
            }

        };

    }

    template<> struct Fingerprint<Identity> : Detail::FingerprintRecord<1> {};
    template<> struct Fingerprint<Affine  > : Detail::FingerprintRecord<4> {};
    template<> struct Fingerprint<Concat  > : Detail::FingerprintRecord<8> {};

    template<i32 L, i32 H>
    struct Fingerprint<ClippedReLU<L, H>> : Detail::FingerprintBounds<2, L, H> {};

    template<i32 L, i32 H>
    struct Fingerprint<SquaredClippedReLU<L, H>> : Detail::FingerprintBounds<3, L, H> {};

    template<typename T> struct Fingerprint<Accumulate<T>> : Detail::FingerprintRecord< 6, T> {};
    template<typename T> struct Fingerprint<Mirror    <T>> : Detail::FingerprintRecord< 7, T> {};
    template<typename T> struct Fingerprint<Residual  <T>> : Detail::FingerprintRecord<10, T> {};

    template<typename... T> struct Fingerprint<Parallel<T...>> : Detail::FingerprintList< 9, T...> {};
    template<typename... T> struct Fingerprint<Sequence<T...>> : Detail::FingerprintList<11, T...> {};

    template<s00 I, s00 O, typename A, typename T>
    struct Fingerprint<Layer<I, O, A, T>>
    {

        constexpr static void Append(Detail::FingerprintState& state)
        {
            state.Add(5);
            state.Add(I);
            state.Add(O);
            Fingerprint<A>::Append(state);
            Fingerprint<T>::Append(state);
        }

    };

    template<i32 A, i32 B, i32 Scale, typename F, typename W, typename S>
    struct Fingerprint<Quantization<A, B, Scale, F, W, S>>
    {

        constexpr static void Append(Detail::FingerprintState& state)
        {
            for (const u64 value : std::array<u64, 7> { 12, A, B, Scale, sizeof(F), sizeof(W), sizeof(S) })
                state.Add(value);
        }

    };

    template<typename Q, typename... N>
    struct Fingerprint<Network<Q, N...>>
    {

        constexpr static void Append(Detail::FingerprintState& state)
        {
            state.Add(13);
            Fingerprint<Q>::Append(state);
            state.Add(sizeof...(N));
            (Fingerprint<N>::Append(state), ...);
        }

    };

    template<typename Architecture>
    consteval u64 ArchitectureFingerprint()
    {
        Detail::FingerprintState state;

        Fingerprint<Architecture>::Append(state);

        return state.Value;
    }

    namespace Detail
    {

        constexpr inline std::array Magic {
            static_cast<std::byte>('M'), static_cast<std::byte>('A'), static_cast<std::byte>('N'),
            static_cast<std::byte>('T'), static_cast<std::byte>('A'), static_cast<std::byte>('R'),
            static_cast<std::byte>('A'), static_cast<std::byte>('Y')
        };

        template<typename Stream>
        bool ReadBytes(Stream& stream, const std::span<std::byte> bytes)
        {
            if constexpr (requires { { stream.ReadBytes(bytes) } -> std::same_as<bool>; })
                return stream.ReadBytes(bytes);
            else {
                if (bytes.size() > static_cast<s00>(std::numeric_limits<std::streamsize>::max())) return false;

                stream.read(
                    reinterpret_cast<      char*    >(bytes.data()),
                         static_cast<std::streamsize>(bytes.size())
                );

                return static_cast<bool>(stream);
            }
        }

        template<typename Stream>
        bool WriteBytes(Stream& stream, const std::span<const std::byte> bytes)
        {
            if constexpr (requires { { stream.WriteBytes(bytes) } -> std::same_as<bool>; })
                return stream.WriteBytes(bytes);
            else {
                if (bytes.size() > static_cast<s00>(std::numeric_limits<std::streamsize>::max())) return false;

                stream.write(
                    reinterpret_cast<  const char*  >(bytes.data()),
                         static_cast<std::streamsize>(bytes.size())
                );

                return static_cast<bool>(stream);
            }
        }

        template<typename Stream>
        bool Flush(Stream& stream)
        {
            if constexpr (requires { { stream.Flush() } -> std::same_as<bool>; })
                return stream.Flush();
            else if constexpr (requires { stream.flush(); }) {
                stream.flush();

                return static_cast<bool>(stream);
            } else return true;
        }

        template<typename T>
        struct PackedBytes { constexpr static s00 Value = sizeof(T); };

        template<typename T, s00 N>
        struct PackedBytes<std::array<T, N>>
        {

            constexpr static s00 Value = N * PackedBytes<T>::Value;

            static_assert(Value == sizeof(std::array<T, N>));

        };

        template<typename T>
        void SwapBytes(T& tensor)
        {
            if constexpr (std::integral<T>) tensor = std::byteswap(tensor);
            else for (auto& element : tensor) SwapBytes(element);
        }

        template<typename Stream, typename T>
        bool ReadTensor(Stream& stream, T& tensor)
        {
            static_assert(sizeof(T) == PackedBytes<T>::Value);
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);

            if (!ReadBytes(stream, std::as_writable_bytes(std::span(&tensor, 1)))) return false;

            if (std::endian::native == std::endian::big) SwapBytes(tensor);

            return true;
        }

        template<typename Stream, typename T>
        bool WriteTensor(Stream& stream, const T& tensor)
        {
            static_assert(sizeof(T) == PackedBytes<T>::Value);
            static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);

            if (std::endian::native == std::endian::little)
                return WriteBytes(stream, std::as_bytes(std::span(&tensor, 1)));

            auto converted = std::make_unique<T>(tensor);
            SwapBytes(*converted);

            return WriteBytes(stream, std::as_bytes(std::span(converted.get(), 1)));
        }

        template<typename Storage>
        u64 ParameterBytes(const Storage& storage)
        {
            u64 bytes = 0;

            storage.VisitParameters([&]<typename T0>(const T0&) {
                bytes += PackedBytes<std::remove_cvref_t<T0>>::Value;
            });

            return bytes;
        }

        template<typename A>
        struct LegacyV2 : std::false_type {};

        template<typename Q, s00 I, s00 H, typename A, s00 D, s00 O, typename B>
        struct LegacyV2<Network<Q, Mirror<Accumulate<Layer<I, H, A>>>, Concat, Layer<D, O, B>>> :
        std::bool_constant<
            std::same_as<typename Q::FeatureType, i16> &&
            std::same_as<typename Q:: WeightType, i16> &&
            std::same_as<typename Q::    SumType, i32> &&
            D == H * 2
        > {};

    }

}

#endif
