//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BINARYMEMORYSTREAM_H
#define MANTARAY_BINARYMEMORYSTREAM_H

#include <istream>
#include <streambuf>

namespace MantaRay
{

    using StreamBuffer = std::basic_streambuf<char>;

    struct BinaryMemoryBuffer : StreamBuffer
    {

        BinaryMemoryBuffer(const char* src, const usize size)
        {
            auto* p (const_cast<char*>(src));
            this->setg(p, p, p + size);
        }

    };

    using ReadOnlyStream = std::basic_istream<char>;

    struct BinaryMemoryStream final : virtual BinaryMemoryBuffer, ReadOnlyStream
    {

        BinaryMemoryStream(const unsigned char* src, const usize size) :
            BinaryMemoryBuffer(reinterpret_cast<const char*>(src), size),
            ReadOnlyStream(static_cast<StreamBuffer*>(this)) {}

        template<typename T, usize Size>
        void ReadArray(std::array<T, Size>& array)
        {
            this->read(reinterpret_cast<char*>(&array), sizeof array);
        }

    };

} // MantaRay

#endif //MANTARAY_BINARYMEMORYSTREAM_H
