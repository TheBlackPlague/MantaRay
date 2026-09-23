//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#pragma once

#include <tuple>

namespace MantaRay { template<typename... T> struct Sequence { using Nodes = std::tuple<T...>; }; }
