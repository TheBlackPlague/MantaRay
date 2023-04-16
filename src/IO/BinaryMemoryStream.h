//
// Copyright (c) 2023 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_BINARYMEMORYSTREAM_H
#define MANTARAY_BINARYMEMORYSTREAM_H

#include <streambuf>
#include <istream>
#include <array>
#include <cstdint>

#include "DataStream.h"

namespace MantaRay
{

    struct BinaryMemoryBuffer : std::streambuf
    {

        public:
            BinaryMemoryBuffer(const char* src, const size_t size) {
                char *p (const_cast<char*>(src));
                this->setg(p, p, p + size);
            }

    };

    struct BinaryMemoryStream : virtual MantaRay::BinaryMemoryBuffer, std::istream
    {

        public:
            BinaryMemoryStream(const char* src, const size_t size) :
                    BinaryMemoryBuffer(src, size), std::istream(static_cast<std::streambuf*>(this)) {}

            template<typename T, size_t Size>
            void ReadArray(std::array<T, Size> &array)
            {
                this->read((char*)(&array), sizeof array);
            }

    };

} // MantaRay

#endif //MANTARAY_BINARYMEMORYSTREAM_H
