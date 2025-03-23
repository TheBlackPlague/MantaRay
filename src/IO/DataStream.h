//
// Copyright (c) 2023 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_DATASTREAM_H
#define MANTARAY_DATASTREAM_H

#include <fstream>
#include <ios>

namespace MantaRay
{

    /// \brief A wrapper around std::fstream.
    /// \tparam Mode The open mode of the stream.
    /// \details This class is a wrapper around std::fstream. It is used to read and write data to files. It is not
    ///          intended to be used directly but instead through specific implementations.
    template<std::ios_base::openmode Mode>
    class DataStream
    {

        using FileStream = std::fstream;

        protected:
        FileStream Stream;

        public:
        /// \brief The Wrapper Constructor.
        /// \param path The path to the file.
        /// \details This constructor is used to open a file stream to the specified path.
        explicit DataStream(const std::string& path)
        {
            Stream.open(path, Mode);
        }

        /// \brief The DataStream Destructor.
        /// \details This destructor closes the file stream.
        virtual ~DataStream()
        {
            Stream.close();
        }

    };

} // MantaRay

#endif //MANTARAY_DATASTREAM_H
