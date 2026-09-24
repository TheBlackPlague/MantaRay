//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef MANTARAY_IO_WRITER_H
#define MANTARAY_IO_WRITER_H

#include "Format.h"

namespace MantaRay::IO
{

    template<typename Stream, typename Architecture>
    [[nodiscard]]
    bool Write(Stream& stream, const Backend::NetworkStorage<Architecture>& source)
    {
        constexpr u32 version = 4;

        constexpr u32 reserved = 0;

        constexpr u64 fingerprint = ArchitectureFingerprint<Architecture>();

        const u64 bytes = Detail::ParameterBytes(source);
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
