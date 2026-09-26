//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_BINARYMEMORYSTREAM_H
#define MANTARAY_IO_BINARYMEMORYSTREAM_H

#include <cstring>
#include <span>

#include "../Common/Integral.h"

namespace MantaRay
{

    class BinaryMemoryStream
    {

        std::span<const std::byte> Bytes;

        s00 Cursor = 0;

        bool Valid = true;

        public:
        explicit BinaryMemoryStream(const std::span<const std::byte> bytes) : Bytes(bytes) {}

        BinaryMemoryStream(const u08* bytes, const s00 count) :
            Bytes(reinterpret_cast<const std::byte*>(bytes), count) {}

        BinaryMemoryStream(const char* bytes, const s00 count) :
            Bytes(reinterpret_cast<const std::byte*>(bytes), count) {}

        [[nodiscard]]
        bool ReadBytes(const std::span<std::byte> destination)
        {
            Valid = Valid && destination.size() <= Remaining();

            if (!Valid) return false;

            if (!destination.empty()) std::memcpy(destination.data(), Bytes.data() + Cursor, destination.size());

            Cursor += destination.size();

            return true;
        }

        [[nodiscard]]
        bool ReadBytes(void* destination, const s00 count)
        { return ReadBytes(std::span(static_cast<std::byte*>(destination), count)); }

        [[nodiscard]]
        bool Good() const { return Valid; }

        [[nodiscard]]
        s00 Remaining() const { return Bytes.size() - Cursor; }

        [[nodiscard]]
        s00 Offset() const { return Cursor; }

    };

}

#endif
