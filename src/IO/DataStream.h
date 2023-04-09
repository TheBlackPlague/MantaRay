//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_DATASTREAM_H
#define CEREBRUM_DATASTREAM_H

#include <iostream>
#include <fstream>
#include <ios>

#define FileStream std::fstream

namespace Cerebrum
{

    template<std::ios_base::openmode O>
    class DataStream
    {

        protected:
            FileStream Stream;

        public:
            explicit DataStream(const std::string &path)
            {
                Stream.open(path, O);
            }
    };

} // Cerebrum

#endif //CEREBRUM_DATASTREAM_H
