//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_BINARYSTREAM_H
#define CEREBRUM_BINARYSTREAM_H

#include <array>
#include <cstdint>
#include "DataStream.h"

namespace Cerebrum
{

    class BinaryStream : DataStream<std::ios::binary | std::ios::in>
    {

        private:
            std::string Path;

        public:
            __attribute__((unused)) explicit BinaryStream(const std::string &path) : DataStream(path)
            {
                Path = path;
            }

            void WriteMode()
            {
                // Weird C++ issue workaround.
                this->Stream.close();
                this->Stream.open(Path, std::ios::binary | std::ios::out);
            }

            template<typename T, size_t Size>
            void ReadArray(std::array<T, Size> &array)
            {
                this->Stream.read((char*)(&array), sizeof array);
            }

            template<typename T, size_t Size>
            void WriteArray(const std::array<T, Size> &array)
            {
                this->Stream.write((const char*)(&array), sizeof array);
            }

    };

} // Cerebrum

#endif //CEREBRUM_BINARYSTREAM_H
