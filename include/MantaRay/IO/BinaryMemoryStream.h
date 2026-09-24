//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_IO_BINARYMEMORYSTREAM_H
#define MANTARAY_IO_BINARYMEMORYSTREAM_H

#include <cstddef>
#include <cstring>
#include <span>

#include "../Common/Integral.h"

namespace MantaRay
{

    class BinaryMemoryStream
    {

        std::span<const std::byte> Data;

        s00 Position = 0;

        bool Valid = true;

        public:
        explicit BinaryMemoryStream(const std::span<const std::byte> data) : Data(data) {}

        BinaryMemoryStream(const  u08* data, const s00 size) :
            Data(reinterpret_cast<const std::byte*>(data), size) {}

        BinaryMemoryStream(const char* data, const s00 size) :
            Data(reinterpret_cast<const std::byte*>(data), size) {}

        [[nodiscard]]
        bool ReadBytes(const std::span<std::byte> destination)
        {
            if (!Valid || destination.size() > Remaining()) {
                Valid = false;
                return false;
            }

            if (!destination.empty()) std::memcpy(destination.data(), Data.data() + Position, destination.size());

            Position += destination.size();

            return true;
        }

        [[nodiscard]]
        bool ReadBytes(void* destination, const s00 size)
        {
            return ReadBytes({ static_cast<std::byte*>(destination), size });
        }

        [[nodiscard]]
        bool Good() const { return Valid; }

        [[nodiscard]]
        s00 Remaining() const { return Data.size() - Position; }

        [[nodiscard]]
        s00 Offset() const { return Position; }

    };

}

#endif
