//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_CLIPPEDRELU_H
#define CEREBRUM_CLIPPEDRELU_H

#include <utility>
#include "../Backend/Avx.h"
#include "../Backend/Avx2.h"

namespace Cerebrum
{

    template<typename T, T Minimum, T Maximum>
    class ClippedReLU
    {

        public:
            static inline Vec256I Activate(Vec256I arg)
            {
                const Vec256I min = Avx<T>::From(Minimum);
                const Vec256I max = Avx<T>::From(Maximum);

                return Avx2<T>::Max(min, Avx2<T>::Min(max, arg));
            }

            static inline T Activate(T arg)
            {
                return std::max(Minimum, std::min(Maximum, arg));
            }

    };

} // Cerebrum

#endif //CEREBRUM_CLIPPEDRELU_H
