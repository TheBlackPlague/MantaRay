//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_LIST_H
#define MANTARAY_COMMON_LIST_H

#include <type_traits>

#include "Integral.h"

namespace MantaRay
{

    template<typename... T>
    struct List
    {

        constexpr static s00 Size = sizeof...(T);

    };

    template<s00 I, typename... T>
    using At = __type_pack_element<I, T...>;

    namespace Detail
    {

        template<s00 I, typename... T>
        consteval auto Pick(List<T...>)
        {
            if constexpr (I < sizeof...(T)) return std::type_identity<At<I, T...>> {};
            else                            return std::type_identity<void       > {};
        }

    }

    template<typename L, s00 I>
    using ItemAt = decltype(Detail::Pick<I>(L {}))::type;

}

#endif //MANTARAY_COMMON_LIST_H
