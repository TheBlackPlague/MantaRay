//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#ifdef __ARM_NEON__

#ifndef MANTARAY_NEON_H
#define MANTARAY_NEON_H

#include "ARM64.h"

#include "../Container.h"

namespace MantaRay
{

    template<QuantizedInteger T> struct NeonVec128I;

    template<                  > struct NeonVec128I<i08> { using T = int8x16_t; using E = int16x8_t; };
    template<                  > struct NeonVec128I<i16> { using T = int16x8_t; using E = int32x4_t; };
    template<                  > struct NeonVec128I<i32> { using T = int32x4_t; using E = int64x2_t; };

    template<QuantizedInteger T>
    using Vec128I  = typename NeonVec128I<T>::T;

    template<QuantizedInteger T>
    using Vec128IE = typename NeonVec128I<T>::E;

#ifdef ALIGN
#undef ALIGN
#endif

#define ALIGN alignas(sizeof(MantaRay::Vec128I<MantaRay::i32>))

    template<QuantizedInteger T>
    struct NEON : ARM64<T>
    {

        using Vec128I  = Vec128I <T>;
        using Vec128IE = Vec128IE<T>;

        constexpr static Vec128I Zero = vdupq_n_s64(0);

        static inline Vec128I From(const T value)
        {
            if (std::is_same_v<T, i08>) return vdupq_n_s8 (value);
            if (std::is_same_v<T, i16>) return vdupq_n_s16(value);
            if (std::is_same_v<T, i32>) return vdupq_n_s32(value);

            __builtin_unreachable();
        }

        template<s00 Size>
        static inline Vec128I From(const Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec128I), "Array size must be at least the width of a Vec128I.");

            if (std::is_same_v<T, i08>) return vld1q_s8 (reinterpret_cast<const i08*>(&array[index]));
            if (std::is_same_v<T, i16>) return vld1q_s16(reinterpret_cast<const i16*>(&array[index]));
            if (std::is_same_v<T, i32>) return vld1q_s32(reinterpret_cast<const i32*>(&array[index]));

            __builtin_unreachable();
        }

        template<s00 Size>
        static inline void Store(const Vec128I& q0, Array<T, Size>& array, const s00 index)
        {
            static_assert(sizeof(array) >= sizeof(Vec128I), "Array size must be at least the width of a Vec128I.");

            if (std::is_same_v<T, i08>) vst1q_s8 (reinterpret_cast<i08*>(&array[index]), q0);
            if (std::is_same_v<T, i16>) vst1q_s16(reinterpret_cast<i16*>(&array[index]), q0);
            if (std::is_same_v<T, i32>) vst1q_s32(reinterpret_cast<i32*>(&array[index]), q0);
        }

        static inline Vec128I Min(const Vec128I& q0, const Vec128I& q1)
        {
            if (std::is_same_v<T, i08>) return vminq_s8 (q0, q1);
            if (std::is_same_v<T, i16>) return vminq_s16(q0, q1);
            if (std::is_same_v<T, i32>) return vminq_s32(q0, q1);

            __builtin_unreachable();
        }

        static inline Vec128I Max(const Vec128I& q0, const Vec128I& q1)
        {
            if (std::is_same_v<T, i08>) return vmaxq_s8 (q0, q1);
            if (std::is_same_v<T, i16>) return vmaxq_s16(q0, q1);
            if (std::is_same_v<T, i32>) return vmaxq_s32(q0, q1);

            __builtin_unreachable();
        }

        static inline Vec128I Add(const Vec128I& q0, const Vec128I& q1)
        {
            if (std::is_same_v<T, i08>) return vaddq_s8 (q0, q1);
            if (std::is_same_v<T, i16>) return vaddq_s16(q0, q1);
            if (std::is_same_v<T, i32>) return vaddq_s32(q0, q1);

            __builtin_unreachable();
        }

        static inline Vec128I Sub(const Vec128I& q0, const Vec128I& q1)
        {
            if (std::is_same_v<T, i08>) return vsubq_s8 (q0, q1);
            if (std::is_same_v<T, i16>) return vsubq_s16(q0, q1);
            if (std::is_same_v<T, i32>) return vsubq_s32(q0, q1);

            __builtin_unreachable();
        }

        static inline Vec128IE Madd(const Vec128I& q0, const Vec128I& q1) requires std::is_same_v<T, i16>
        {
            // q0 = [a, b, c, d, e, f, g, h]
            // q1 = [i, j, k, l, m, n, o, p]

            Vec128IE q2;
            Vec128IE q3;
            Vec128IE q4;

            // q2 = [a, b, c, d]
            q2 = vget_low_s16(q0);

            // q3 = [i, j, k, l]
            q3 = vget_low_s16(q1);

            //     [  a  ,   b  ,   c  ,   d  ]
            // *   [  i  ,   j  ,   k  ,   l  ]
            // =   [a * i, b * j, c * k, d * l]
            q2 = vmull_s16(q2, q3);

            // q3 = [e, f, g, h]
            q3 = vget_high_s16(q0);

            // q4 = [m, n, o, p]
            q4 = vget_high_s16(q1);

            //     [  e  ,   f  ,   g  ,   h  ]
            // *   [  m  ,   n  ,   o  ,   p  ]
            // =   [e * m, f * n, g * o, h * p]
            q3 = vmull_s16(q3, q4);

            //     [    a * i    ,     b * j    ,     c * k    ,     d * l    ]
            // +   [    e * m    ,     f * n    ,     g * o    ,     h * p    ]
            // =   [a * i + e * m, b * j + f * n, c * k + g * o, d * l + h * p]
            return vadd_s32(q2, q3);
        }

        static inline T Sum(const Vec128I& q0) requires std::is_same_v<T, i32>
        {
            // q0 = [a, b, c, d]

            Vec128I q1;

            T w0;
            T w1;

            // q1 = [c, d, c, d]
            q1 = vextq_s32(q0, q0, 2);

            //     [  a  ,   b  ,   c  ,   d  ]
            // +   [  c  ,   d  ,   c  ,   d  ]
            // =   [a + c, b + d, c + c, d + d]
            q1 = Add(q0, q1);

            // w0 = a + c
            w0 = vgetq_lane_s32(q1, 0);

            // w1 = b + d
            w1 = vgetq_lane_s32(q1, 1);

            // w0 + w1 = (a + c) + (b + d) = a + c + b + d = a + b + c + d
            return w0 + w1;
        }

    };

} // MantaRay

#endif //MANTARAY_NEON_H

#endif
