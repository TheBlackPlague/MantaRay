//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_BINARYFILESTREAM_H
#define MANTARAY_IO_BINARYFILESTREAM_H

#include <filesystem>
#include <fstream>
#include <limits>
#include <span>

#include "../Common/Integral.h"

namespace MantaRay
{

    template<bool Read = true>
    class BinaryFileStream
    {

        std::fstream File;

        bool Fits(const s00 count)
        {
            if (count <= static_cast<s00>(std::numeric_limits<std::streamsize>::max())) return true;

            File.setstate(std::ios::failbit);

            return false;
        }

        public:
        explicit BinaryFileStream(const std::filesystem::path& path) :
            File(path, std::ios::binary | (Read ? std::ios::in : std::ios::out | std::ios::trunc)) {}

        [[nodiscard]]
        bool ReadBytes(const std::span<std::byte> bytes) requires Read
        {
            if (!Fits(bytes.size())) return false;

            if (!bytes.empty()) File.read(
                reinterpret_cast<      char*    >(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())
            );

            return Good();
        }

        [[nodiscard]]
        bool ReadBytes(void* destination, const s00 count) requires Read
        { return ReadBytes(std::span(static_cast<std::byte*>(destination), count)); }

        [[nodiscard]]
        bool WriteBytes(const std::span<const std::byte> bytes) requires (!Read)
        {
            if (!Fits(bytes.size())) return false;

            if (!bytes.empty()) File.write(
                reinterpret_cast<  const char*  >(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())
            );

            return Good();
        }

        [[nodiscard]]
        bool WriteBytes(const void* source, const s00 count) requires (!Read)
        { return WriteBytes(std::span(static_cast<const std::byte*>(source), count)); }

        [[nodiscard]]
        bool Flush() requires (!Read)
        {
            File.flush();

            return Good();
        }

        [[nodiscard]]
        bool Good() const { return static_cast<bool>(File); }

    };

}

#endif
