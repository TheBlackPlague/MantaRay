//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_IO_READER_H
#define MANTARAY_IO_READER_H

#include <memory>
#include <tuple>
#include <utility>

#include "Format.h"

namespace MantaRay::IO
{

    template<typename Stream, typename Architecture>
    [[nodiscard]]
    bool Read(Stream& stream, Backend::NetworkStorage<Architecture>& destination)
    {
        std::array<std::byte, 8> magic {};

        u32 version     = 0;
        u32 reserved    = 0;
        u64 fingerprint = 0;

        u64 bytes = 0;
        if (!Detail::ReadBytes (stream, magic      ) || magic != Detail::Magic                                 ||
            !Detail::ReadTensor(stream, version    ) || version != 4                                           ||
            !Detail::ReadTensor(stream, reserved   ) || reserved != 0                                          ||
            !Detail::ReadTensor(stream, fingerprint) || fingerprint != ArchitectureFingerprint<Architecture>() ||
            !Detail::ReadTensor(stream, bytes      ) || bytes != Detail::ParameterBytes(destination))
            return false;

        auto loaded = std::make_unique<Backend::NetworkStorage<Architecture>>();

        bool valid = true;

        loaded->VisitParameters([&](auto& parameter) {
            if (valid) valid = Detail::ReadTensor(stream, parameter);
        });

        if (!valid) return false;

        destination = std::move(*loaded);

        return true;
    }

    template<typename Stream, typename Architecture> requires Detail::LegacyV2<Architecture>::value
    [[nodiscard]]
    bool ReadLegacyV2(Stream& stream, Backend::NetworkStorage<Architecture>& destination)
    {
        auto loaded = std::make_unique<Backend::NetworkStorage<Architecture>>();

        auto& feature = std::get<0>(loaded->Nodes).Inner;
        auto& dense   = std::get<2>(loaded->Nodes)      ;

        std::array<i16, std::tuple_size_v<decltype(dense.Bias)>> bias {};

        if (!Detail::ReadTensor(stream, feature.Weight) || !Detail::ReadTensor(stream, feature.Bias) ||
            !Detail::ReadTensor(stream,   dense.Weight) || !Detail::ReadTensor(stream,         bias)  )
            return false;

        for (s00 i = 0; i < bias.size(); i++) dense.Bias[i] = bias[i];

        destination = std::move(*loaded);

        return true;
    }

}

#endif
