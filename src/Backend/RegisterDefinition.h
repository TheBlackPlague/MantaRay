//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifndef CEREBRUM_REGISTERDEFINITION_H
#define CEREBRUM_REGISTERDEFINITION_H

#include "immintrin.h"

#ifdef __AVX512F__
using Vec512I = __m512i;
using Vec256I = __m256i;
using Vec128I = __m128i;
#elif __AVX__
using Vec256I = __m256i;
using Vec128I = __m128i;
#elif __SSE__
using Vec128I = __m128i;
#endif

#endif //CEREBRUM_REGISTERDEFINITION_H
