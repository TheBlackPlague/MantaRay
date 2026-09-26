//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_READER_H
#define MANTARAY_IO_READER_H

#include "Format.h"

namespace MantaRay::IO
{

    template<typename Stream, typename Architecture>
    [[nodiscard]]
    bool Read(Stream& stream, Backend::NetworkStorage<Architecture>& destination)
    {
        std::array<std::byte, 8> magic {};

        u32 version   {}, reserved {};
        u64 signature {}, length   {};

        if (!Detail::ReadBytes (stream,     magic) || magic     !=                           Detail::Magic)
            return false;
        if (!Detail::ReadTensor(stream,   version) || version   !=                                       4)
            return false;
        if (!Detail::ReadTensor(stream,  reserved) || reserved  !=                                       0)
            return false;
        if (!Detail::ReadTensor(stream, signature) || signature != ArchitectureFingerprint<Architecture>())
            return false;
        if (!Detail::ReadTensor(stream,    length) || length    !=     Detail::ParameterBytes(destination))
            return false;

        auto replacement = std::make_unique<Backend::NetworkStorage<Architecture>>();
        bool success = true;

        replacement->VisitParameters([&](auto& tensor) { success = success && Detail::ReadTensor(stream, tensor); });
        if (success) destination = std::move(*replacement);

        return success;
    }

    template<typename Stream, typename Architecture>
    requires Detail::LegacyV2<Architecture>::value
    [[nodiscard]]
    bool ReadLegacyV2(Stream& stream, Backend::NetworkStorage<Architecture>& destination)
    {
        auto replacement = std::make_unique<Backend::NetworkStorage<Architecture>>();

        auto& feature = std::get<0>(replacement->Nodes()).Inner;
        auto& output  = std::get<2>(replacement->Nodes())      ;

        std::array<i16, std::tuple_size_v<decltype(output.Bias)>> oldBias {};

        const bool success = Detail::ReadTensor(stream, feature.Weight) && Detail::ReadTensor(stream, feature.Bias) &&
                             Detail::ReadTensor(stream,  output.Weight) && Detail::ReadTensor(stream,      oldBias)  ;

        if (!success) return false;

        for (s00 index = 0; index != oldBias.size(); index++) output.Bias[index] = oldBias[index];

        destination = std::move(*replacement);

        return true;
    }

}

#endif
