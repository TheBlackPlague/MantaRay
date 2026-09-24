//
// Copyright (c) 2025-2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_RUNTIME_STATE_H
#define MANTARAY_RUNTIME_STATE_H

#include "../Backend/Storage/Accumulator.h"

namespace MantaRay::Runtime
{

    template<typename Architecture>
    using State = Backend::Accumulator<Architecture>;

}

#endif
