#ifndef MANTARAY_IO_WRITER_H
#define MANTARAY_IO_WRITER_H

#include "Format.h"

namespace MantaRay::IO
{

    template<typename Stream, typename Architecture>
    [[nodiscard]]
    bool Write(Stream& stream, const Backend::NetworkStorage<Architecture>& source)
    {
        constexpr std::uint32_t version = 4;

        constexpr std::uint32_t reserved = 0;

        constexpr std::uint64_t fingerprint = ArchitectureFingerprint<Architecture>();

        const std::uint64_t bytes = Detail::ParameterBytes(source);
        if (!Detail::WriteBytes (stream, Detail::Magic) ||
            !Detail::WriteTensor(stream, version      ) ||
            !Detail::WriteTensor(stream, reserved     ) ||
            !Detail::WriteTensor(stream, fingerprint  ) ||
            !Detail::WriteTensor(stream, bytes        )  )
            return false;

        bool valid = true;

        source.VisitParameters([&](const auto& parameter) {
            if (valid) valid = Detail::WriteTensor(stream, parameter);
        });

        return valid && Detail::Flush(stream);
    }

}

#endif
