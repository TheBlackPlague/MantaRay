#ifndef MANTARAY_IO_BINARYMEMORYSTREAM_H
#define MANTARAY_IO_BINARYMEMORYSTREAM_H

#include <cstddef>
#include <cstring>
#include <span>

namespace MantaRay
{

    class BinaryMemoryStream
    {

        std::span<const std::byte> Data;

        std::size_t Position = 0;

        bool Valid = true;

        public:
        explicit BinaryMemoryStream(const std::span<const std::byte> data) : Data(data) {}

        BinaryMemoryStream(const unsigned char* data, const std::size_t size) :
            Data(reinterpret_cast<const std::byte*>(data), size) {}

        BinaryMemoryStream(const          char* data, const std::size_t size) :
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
        bool ReadBytes(void* destination, const std::size_t size)
        {
            return ReadBytes({static_cast<std::byte*>(destination), size});
        }

        [[nodiscard]]
        bool Good() const { return Valid; }

        [[nodiscard]]
        std::size_t Remaining() const { return Data.size() - Position; }

        [[nodiscard]]
        std::size_t Offset() const { return Position; }

    };

}

#endif
