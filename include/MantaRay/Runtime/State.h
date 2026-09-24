//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
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
