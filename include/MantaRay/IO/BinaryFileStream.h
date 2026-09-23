#ifndef MANTARAY_IO_BINARYFILESTREAM_H
#define MANTARAY_IO_BINARYFILESTREAM_H

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>

namespace MantaRay
{

    template<bool Read = true>
    class BinaryFileStream
    {

        std::fstream Stream;

        public:
        explicit BinaryFileStream(const std::filesystem::path& path) :
            Stream(path, std::ios::binary | (Read ? std::ios::in : std::ios::out | std::ios::trunc)) {}

        [[nodiscard]]
        bool ReadBytes(const std::span<std::byte> destination) requires Read
        {
            if (destination.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                Stream.setstate(std::ios::failbit);
                return false;
            }

            if (!destination.empty()) Stream.read(
                reinterpret_cast<      char*    >(destination.data()),
                     static_cast<std::streamsize>(destination.size())
            );

            return Good();
        }

        [[nodiscard]]
        bool ReadBytes(void* destination, const std::size_t size) requires Read
        {
            return ReadBytes({static_cast<std::byte*>(destination), size});
        }

        [[nodiscard]]
        bool WriteBytes(const std::span<const std::byte> source) requires (!Read)
        {
            if (source.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                Stream.setstate(std::ios::failbit);
                return false;
            }

            if (!source.empty()) Stream.write(
                reinterpret_cast<  const char*  >(source.data()),
                     static_cast<std::streamsize>(source.size())
            );

            return Good();
        }

        [[nodiscard]]
        bool WriteBytes(const void* source, const std::size_t size) requires (!Read)
        {
            return WriteBytes({static_cast<const std::byte*>(source), size});
        }

        [[nodiscard]]
        bool Flush() requires (!Read)
        {
            Stream.flush();
            return Good();
        }

        [[nodiscard]]
        bool Good() const { return static_cast<bool>(Stream); }

    };

}

#endif
