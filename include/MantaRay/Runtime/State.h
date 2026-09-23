#pragma once

#include "../Backend/Storage/Accumulator.h"

namespace MantaRay::Runtime
{

    template<class Architecture>
    using State = Backend::Accumulator<Architecture>;

}
