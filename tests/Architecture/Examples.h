//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_EXAMPLES_H
#define MANTARAY_ARCHITECTURE_EXAMPLES_H

#include <MantaRay/MantaRay.h>

namespace Examples
{

    using namespace MantaRay;

    template<s00 Hidden>
    constexpr auto Perspective()
    {
        const auto [stm, ntm] = Input<Chess768>();

        const auto l0 = Affine(768       , Hidden, Weights<i16>(255, Clip(1.98)), Biases<i16>(255, Clip(1.98)));
        const auto l1 = Affine(Hidden * 2, 1     , Weights<i16>(64 , Clip(1.98)), Biases<i16>(255 * 64       ));

        const auto y0 = ClippedReLU(l0(stm));
        const auto y1 = ClippedReLU(l0(ntm));

        return Output(l1(Concat(y0, y1)), 400);
    }

    constexpr auto Starshard() { return Perspective<256>(); }
    constexpr auto Aurora   () { return Perspective<384>(); }
    constexpr auto Flare    () { return Perspective<768>(); }

}

#endif //MANTARAY_ARCHITECTURE_EXAMPLES_H
