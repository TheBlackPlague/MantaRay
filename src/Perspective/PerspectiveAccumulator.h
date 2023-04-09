//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_PERSPECTIVEACCUMULATOR_H
#define CEREBRUM_PERSPECTIVEACCUMULATOR_H

#include <array>

namespace Cerebrum
{

    template<typename T, size_t AccumulatorSize>
    class PerspectiveAccumulator
    {

        public:
            std::array<T, AccumulatorSize> White;
            std::array<T, AccumulatorSize> Black;

            PerspectiveAccumulator()
            {
                std::fill(std::begin(White), std::end(White), 0);
                std::fill(std::begin(Black), std::end(Black), 0);
            }

            void CopyTo(PerspectiveAccumulator<T, AccumulatorSize> &accumulator)
            {
                std::copy(std::begin(White), std::end(White), std::begin(accumulator.White));
                std::copy(std::begin(Black), std::end(Black), std::begin(accumulator.Black));
            }

            void LoadBias(std::array<T, AccumulatorSize> &bias)
            {
                std::copy(std::begin(bias), std::end(bias), std::begin(White));
                std::copy(std::begin(bias), std::end(bias), std::begin(Black));
            }

            void Zero()
            {
                std::fill(std::begin(White), std::end(White), 0);
                std::fill(std::begin(Black), std::end(Black), 0);
            }

    };

} // Cerebrum

#endif //CEREBRUM_PERSPECTIVEACCUMULATOR_H
