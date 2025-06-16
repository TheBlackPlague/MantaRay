//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BINARYFILESTREAM_H
#define MANTARAY_BINARYFILESTREAM_H

#include <fstream>
#include <ios>

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

        template<typename T, usize N>
        void ReadArray(Array<T, N>& dst) requires Read
        {
            Stream.read(reinterpret_cast<char*>(&dst), sizeof dst);
        }

        template<typename T, usize N>
        void WriteArray(const Array<T, N>& src) requires (!Read)
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
