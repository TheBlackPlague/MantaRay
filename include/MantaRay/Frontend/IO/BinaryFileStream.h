//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BINARYFILESTREAM_H
#define MANTARAY_BINARYFILESTREAM_H

#include <fstream>
#include <ios>
#include <type_traits>

#include "../../Backend/Container.h"

namespace MantaRay
{

    using FileStream = std::fstream;

    template<bool Read = true>
    class BinaryFileStream
    {

        FileStream Stream;

        public:
        BinaryFileStream(const std::string& path)
        {
            Stream.open(path, std::ios::binary | (Read ? std::ios::in : std::ios::out));
        }

        template<typename T, s00 N>
        void ReadArray(std::array<T, N>& dst) requires (Read && std::is_trivially_copyable_v<std::array<T, N>>)
        {
            Stream.read(reinterpret_cast<char*>(&dst), sizeof dst);
        }

        template<typename T, s00 N>
        void WriteArray(const std::array<T, N>& src) requires (!Read && std::is_trivially_copyable_v<std::array<T, N>>)
        {
            Stream.write(reinterpret_cast<const char*>(&src), sizeof src);
        }

        ~BinaryFileStream()
        {
            Stream.close();
        }

    };

} // MantaRay

#endif //MANTARAY_BINARYFILESTREAM_H
