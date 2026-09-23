//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#pragma once

namespace MantaRay
{

    template<int Min = 0, int Max = 1>
    struct SquaredClippedReLU
    {

        static constexpr int Minimum = Min;
        static constexpr int Maximum = Max;

    };

}
