#ifndef MANTARAY_IO_FORMAT_H
#define MANTARAY_IO_FORMAT_H

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <ios>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "../Architecture/Network.h"

namespace MantaRay::IO
{

    namespace Detail
    {

        struct FingerprintState
        {

            std::uint64_t Value = 14695981039346656037ULL;

            constexpr void Add(std::uint64_t value)
            {
                for (unsigned i = 0; i < 8; ++i) {
                    Value = (Value ^ (value & 255)) * 1099511628211ULL;
                    value >>= 8;
                }
            }

        };

    }

    template<typename T>
    struct Fingerprint;

    template<>
    struct Fingerprint<Identity>
    {

        static constexpr void Append(Detail::FingerprintState& hash) { hash.Add(1); }

    };

    template<int Minimum, int Maximum>
    struct Fingerprint<ClippedReLU<Minimum, Maximum>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(2); hash.Add(Minimum); hash.Add(Maximum);
        }

    };

    template<int Minimum, int Maximum>
    struct Fingerprint<SquaredClippedReLU<Minimum, Maximum>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(3); hash.Add(Minimum); hash.Add(Maximum);
        }

    };

    template<>
    struct Fingerprint<Affine>
    {

        static constexpr void Append(Detail::FingerprintState& hash) { hash.Add(4); }

    };

    template<std::size_t Input, std::size_t Output, typename Activation, typename Transform>
    struct Fingerprint<Layer<Input, Output, Activation, Transform>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(5);

            hash.Add( Input);
            hash.Add(Output);

            Fingerprint<Activation>::Append(hash);
            Fingerprint<Transform >::Append(hash);
        }

    };

    template<typename T>
    struct Fingerprint<Accumulate<T>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(6); Fingerprint<T>::Append(hash);
        }

    };

    template<typename T>
    struct Fingerprint<Mirror<T>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(7); Fingerprint<T>::Append(hash);
        }

    };

    template<>
    struct Fingerprint<Concat>
    {
        static constexpr void Append(Detail::FingerprintState& hash) { hash.Add(8); }
    };

    template<typename... T>
    struct Fingerprint<Parallel<T...>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(9); hash.Add(sizeof...(T)); (Fingerprint<T>::Append(hash), ...);
        }

    };

    template<typename T>
    struct Fingerprint<Residual<T>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(10); Fingerprint<T>::Append(hash);
        }

    };

    template<typename... T>
    struct Fingerprint<Sequence<T...>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(11); hash.Add(sizeof...(T)); (Fingerprint<T>::Append(hash), ...);
        }

    };

    template<int A, int B, int Scale, typename Feature, typename Weight, typename Sum>
    struct Fingerprint<Quantization<A, B, Scale, Feature, Weight, Sum>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(12); hash.Add(A); hash.Add(B); hash.Add(Scale);
            hash.Add(sizeof(Feature)); hash.Add(sizeof(Weight)); hash.Add(sizeof(Sum));
        }

    };

    template<typename Q, typename... T>
    struct Fingerprint<Network<Q, T...>>
    {

        static constexpr void Append(Detail::FingerprintState& hash)
        {
            hash.Add(13); Fingerprint<Q>::Append(hash);
            hash.Add(sizeof...(T)); (Fingerprint<T>::Append(hash), ...);
        }

    };

    template<typename Architecture>
    consteval std::uint64_t ArchitectureFingerprint()
    {

        Detail::FingerprintState hash;

        Fingerprint<Architecture>::Append(hash);
        return hash.Value;

    }

    namespace Detail
    {

        template<typename Stream>
        bool ReadBytes(Stream& stream, const std::span<std::byte> destination)
        {
            if constexpr (requires { { stream.ReadBytes(destination) } -> std::same_as<bool>; }) {
                return stream.ReadBytes(destination);
            } else {
                if (destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
                    return false;

                stream.read(
                    reinterpret_cast<     char*     >(destination.data()),
                         static_cast<std::streamsize>(destination.size())
                );

                return static_cast<bool>(stream);
            }
        }

        template<typename Stream>
        bool WriteBytes(Stream& stream, const std::span<const std::byte> source)
        {
            if constexpr (requires { { stream.WriteBytes(source) } -> std::same_as<bool>; }) {
                return stream.WriteBytes(source);
            } else {
                if (source.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
                    return false;

                stream.write(
                    reinterpret_cast<  const char*  >(source.data()),
                         static_cast<std::streamsize>(source.size())
                );

                return static_cast<bool>(stream);
            }
        }

        template<typename Stream>
        bool Flush(Stream& stream)
        {
            if        constexpr (requires { { stream.Flush() } -> std::same_as<bool>; }) {
                return stream.Flush();
            } else if constexpr (requires {   stream.flush();                         }) {
                stream.flush();
                return static_cast<bool>(stream);
            } else return true;
        }

        template<typename T>
        struct PackedBytes
        {

            static_assert(std::is_integral_v<T>);

            static constexpr std::size_t Value = sizeof(T);

        };

        template<typename T, std::size_t N>
        struct PackedBytes<std::array<T, N>>
        {

            static constexpr std::size_t Value = N * PackedBytes<T>::Value;

            static_assert(sizeof(std::array<T, N>) == Value, "Parameter arrays must be tightly packed.");

        };

        template<typename T>
        void SwapBytes(T& value)
        {
            if constexpr (std::is_integral_v<T>) value = std::byteswap(value);

            else for (auto& element : value) SwapBytes(element);
        }

        template<typename Stream, typename T>
        bool ReadTensor(Stream& stream, T& destination)
        {
            static_assert(sizeof(T) == PackedBytes<T>::Value);

            if (!ReadBytes(stream, std::as_writable_bytes(std::span(&destination, 1)))) return false;

            if constexpr      (std::endian::native == std::endian::big   ) SwapBytes(destination);
            else static_assert(std::endian::native == std::endian::little, "Mixed-endian targets are unsupported.");

            return true;
        }

        template<typename Stream, typename T>
        bool WriteTensor(Stream& stream, const T& source)
        {
            static_assert(sizeof(T) == PackedBytes<T>::Value);

            if constexpr     (std::endian::native == std::endian::little) {
                return WriteBytes(stream, std::as_bytes(std::span(&source, 1)));
            } else {
                static_assert(std::endian::native == std::endian::big   , "Mixed-endian targets are unsupported.");

                auto swapped = std::make_unique<T>(source);
                SwapBytes(*swapped);

                return WriteBytes(stream, std::as_bytes(std::span(swapped.get(), 1)));
            }
        }

        template<typename Storage>
        std::uint64_t ParameterBytes(const Storage& storage)
        {
            std::uint64_t count = 0;

            storage.VisitParameters([&]<typename T0>(const T0& _) {
                count += PackedBytes<std::remove_cvref_t<T0>>::Value;
            });

            return count;
        }

        inline constexpr std::array Magic = {
            std::byte { 'M' }, std::byte { 'A' }, std::byte { 'N' }, std::byte { 'T' },
            std::byte { 'A' }, std::byte { 'R' }, std::byte { 'A' }, std::byte { 'Y' }
        };

        template<typename _>
        struct LegacyV2 : std::false_type {};

        template<
            typename Q,
            std::size_t Input,
            std::size_t Hidden,
            typename Activation,
            std::size_t DenseInput,
            std::size_t Output,
            typename DenseActivation
        >
        struct LegacyV2<
            Network<
                Q,
                Mirror<
                    Accumulate<
                        Layer<Input, Hidden, Activation>
                    >
                >,
                Concat,
                Layer<DenseInput, Output, DenseActivation>
            >
        > : std::bool_constant<std::is_same_v<typename Q::FeatureType, i16> &&
                               std::is_same_v<typename Q::WeightType , i16> &&
                               std::is_same_v<typename Q::SumType    , i32> && DenseInput == 2 * Hidden> {};

    }

}

#endif
