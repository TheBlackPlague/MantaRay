//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_ACCUMULATOR_H
#define CEREBRUM_ACCUMULATOR_H

#include <array>

namespace Cerebrum
{

    enum AccumulatorOperation
    {
        Activate,
        Deactivate
    };

    template<typename AccumulatorType, size_t AccumulatorSize>
    class Accumulator
    {

        public:
            std::array<AccumulatorType, AccumulatorSize> White;
            std::array<AccumulatorType, AccumulatorSize> Black;

            Accumulator()
            {
                std::fill(std::begin(White), std::end(White), 0);
                std::fill(std::begin(Black), std::end(Black), 0);
            }

            void CopyTo(Accumulator<AccumulatorType, AccumulatorSize> &accumulator)
            {
                std::copy(std::begin(White), std::end(White), std::begin(accumulator.White));
                std::copy(std::begin(Black), std::end(Black), std::begin(accumulator.Black));
            }

            void LoadBias(std::array<AccumulatorType, AccumulatorSize> &bias)
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

#endif //CEREBRUM_ACCUMULATOR_H
