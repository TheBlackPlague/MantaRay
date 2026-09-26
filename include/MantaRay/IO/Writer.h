//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
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
        constexpr u32 version = 4, reserved = 0;
        constexpr u64 signature = ArchitectureFingerprint<Architecture>();

        const u64 length = Detail::ParameterBytes(source);

        bool success = Detail::WriteBytes (stream, Detail::Magic) &&
                       Detail::WriteTensor(stream,       version) &&
                       Detail::WriteTensor(stream,      reserved) &&
                       Detail::WriteTensor(stream,     signature) &&
                       Detail::WriteTensor(stream,        length)  ;

        source.VisitParameters([&](const auto& tensor) { success = success && Detail::WriteTensor(stream, tensor); });

        return success && Detail::Flush(stream);
    }

}

#endif
