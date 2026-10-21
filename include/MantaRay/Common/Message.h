//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_COMMON_MESSAGE_H
#define MANTARAY_COMMON_MESSAGE_H

#include <array>
#include <string_view>

#include "Integral.h"

namespace MantaRay
{

    class Message
    {

        constexpr static s00 Capacity = 256;

        std::array<char, Capacity> Buffer {};

        s00 Length = 0;

        constexpr void Append(const char character)
        {
            if (Length < Capacity - 1) Buffer[Length++] = character;
        }

        public:
        constexpr Message& operator <<(const std::string_view text)
        {
            for (const char character : text) Append(character);

            return *this;
        }

        constexpr Message& operator <<(const i64 value)
        {
            if (value < 0) Append('-');

            u64 rest = value < 0 ? 0 - static_cast<u64>(value) : static_cast<u64>(value);

            std::array<char, 20> digits {};

            s00 count = 0;

            do {
                digits[count++] = static_cast<char>('0' + rest % 10);

                rest /= 10;
            } while (rest != 0);

            while (count != 0) Append(digits[--count]);

            return *this;
        }

        constexpr const char* data() const { return Buffer.data(); }

        constexpr s00 size() const { return Length; }

    };

}

#endif //MANTARAY_COMMON_MESSAGE_H
