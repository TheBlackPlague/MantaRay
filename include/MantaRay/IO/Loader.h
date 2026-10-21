//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_LOADER_H
#define MANTARAY_IO_LOADER_H

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

#include "../Common/Integral.h"
#include "../Common/Interval.h"

namespace MantaRay
{

    enum class LoadStatus : u08 { Ok, SizeMismatch, BadPadding, BoundViolation };

    struct LoadResult
    {

        LoadStatus Status = LoadStatus::Ok;

        u16 Tensor = 0;
        u64 Index  = 0;

        constexpr explicit operator bool() const noexcept { return Status == LoadStatus::Ok; }

        [[nodiscard]]
        constexpr std::string_view Message() const noexcept
        {
            switch (Status) {
                case LoadStatus::Ok            : return "ok"             ;
                case LoadStatus::SizeMismatch  : return "size mismatch"  ;
                case LoadStatus::BadPadding    : return "bad padding"    ;
                case LoadStatus::BoundViolation: return "bound violation";
            }

            return "unknown status";
        }

    };

}

namespace MantaRay::IO
{

    constexpr inline std::string_view Padding      = "bullet";
    constexpr inline s00              PaddingBlock = 64;

    constexpr s00 PaddedSize(const s00 payload) { return (payload + PaddingBlock - 1) / PaddingBlock * PaddingBlock; }

    class Reader
    {

        std::span<const std::byte> Bytes;

        s00 Position = 0;
        u16 Tensors  = 0;

        template<typename T>
        static T FromLittleEndian(const T value)
        {
            if constexpr (std::endian::native == std::endian::little || sizeof(T) == 1) return value;
            else {
                using U = std::conditional_t<sizeof(T) == 2, u16, std::conditional_t<sizeof(T) == 4, u32, u64>>;

                return std::bit_cast<T>(std::byteswap(std::bit_cast<U>(value)));
            }
        }

        public:
        explicit Reader(const std::span<const std::byte> bytes) noexcept : Bytes(bytes) {}

        template<typename T>
        [[nodiscard]]
        bool Read(const std::span<T> values) noexcept
        {
            static_assert(std::is_arithmetic_v<T>, "tensors hold arithmetic values");

            if (values.size_bytes() > Remaining()) return false;

            if (!values.empty()) std::memcpy(values.data(), Bytes.data() + Position, values.size_bytes());

            for (T& value : values) value = FromLittleEndian(value);

            Position += values.size_bytes();
            Tensors++;

            return true;
        }

        [[nodiscard]]
        s00 Remaining() const noexcept { return Bytes.size() - Position; }

        [[nodiscard]]
        u16 Tensor() const noexcept { return Tensors; }

    };

    template<typename Spec, typename R>
    [[nodiscard]]
    LoadResult ReadTensor(R& reader, const std::span<typename Spec::Type> values)
    {
        const u16 tensor = reader.Tensor();

        if (!reader.Read(values)) return { LoadStatus::SizeMismatch, tensor, 0 };

        if constexpr (std::is_integral_v<typename Spec::Type>) {
            for (s00 i = 0; i < values.size(); i++)
                if (values[i] < Spec::Bounds.Min || values[i] > Spec::Bounds.Max)
                    return { LoadStatus::BoundViolation, tensor, i };
        } else {
            for (s00 i = 0; i < values.size(); i++)
                if (!std::isfinite(values[i])) return { LoadStatus::BoundViolation, tensor, i };
        }

        return {};
    }

    [[nodiscard]]
    inline LoadResult CheckSize(const std::span<const std::byte> bytes, const s00 payload, const u16 tensors)
    {
        if (bytes.size() != payload && bytes.size() != PaddedSize(payload))
            return { .Status = LoadStatus::SizeMismatch, .Tensor = tensors, .Index = bytes.size() };

        for (s00 i = payload; i < bytes.size(); i++)
            if (bytes[i] != static_cast<std::byte>(Padding[(i - payload) % Padding.size()]))
                return { .Status = LoadStatus::BadPadding, .Tensor = tensors, .Index = i };

        return {};
    }

}

#endif //MANTARAY_IO_LOADER_H
