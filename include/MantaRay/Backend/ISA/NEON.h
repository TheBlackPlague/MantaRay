//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_ISA_NEON_H
#define MANTARAY_BACKEND_ISA_NEON_H

#if defined(__ARM_NEON) && defined(__aarch64__)

#include <arm_neon.h>

#include "../../Common/Integral.h"

namespace MantaRay::Backend::ISA
{

    class NEON
    {

        [[clang::always_inline]]
        static int32x4_t AsI32(const int16x8_t value) { return vreinterpretq_s32_s16(value); }

        [[clang::always_inline]]
        static int16x8_t AsI16(const int32x4_t value) { return vreinterpretq_s16_s32(value); }

        public:
        using Vector  = int16x8_t  ;
        using VectorF = float32x4_t;

        constexpr static s00 Bytes     = 16;
        constexpr static s00 Registers = 32;
        constexpr static s00 DotChains =  2;

        constexpr static bool VNNI = false;

        [[clang::always_inline]]
        static Vector Load(const void* address) { return vld1q_s16(static_cast<const i16*>(address)); }

        [[clang::always_inline]]
        static void Store(void* address, const Vector value) { vst1q_s16(static_cast<i16*>(address), value); }

        [[clang::always_inline]]
        static Vector Zero() { return vdupq_n_s16(0); }

        [[clang::always_inline]]
        static Vector BroadcastI16(const i16 value) { return vdupq_n_s16(value); }

        [[clang::always_inline]]
        static Vector BroadcastI32(const i32 value) { return AsI16(vdupq_n_s32(value)); }

        [[clang::always_inline]]
        static Vector AddI16(const Vector left, const Vector right) { return vaddq_s16(left, right); }

        [[clang::always_inline]]
        static Vector SubI16(const Vector left, const Vector right) { return vsubq_s16(left, right); }

        [[clang::always_inline]]
        static Vector MinI16(const Vector left, const Vector right) { return vminq_s16(left, right); }

        [[clang::always_inline]]
        static Vector MaxI16(const Vector left, const Vector right) { return vmaxq_s16(left, right); }

        [[clang::always_inline]]
        static Vector MulLoI16(const Vector left, const Vector right) { return vmulq_s16(left, right); }

        [[clang::always_inline]]
        static Vector MulHiI16(const Vector left, const Vector right)
        {
            const auto low  = vmull_s16     (vget_low_s16(left), vget_low_s16(right));
            const auto high = vmull_high_s16(             left ,              right );

            return vuzp2q_s16(AsI16(low), AsI16(high));
        }

        [[clang::always_inline]]
        static Vector AddI32(const Vector left, const Vector right)
        { return AsI16(vaddq_s32(AsI32(left), AsI32(right))); }

        [[clang::always_inline]]
        static Vector SubI32(const Vector left, const Vector right)
        { return AsI16(vsubq_s32(AsI32(left), AsI32(right))); }

        [[clang::always_inline]]
        static Vector MulLoI32(const Vector left, const Vector right)
        { return AsI16(vmulq_s32(AsI32(left), AsI32(right))); }

        [[clang::always_inline]]
        static Vector MaddI16(const Vector left, const Vector right)
        {
            const auto low  = vmull_s16     (vget_low_s16(left), vget_low_s16(right));
            const auto high = vmull_high_s16(             left ,              right );

            return AsI16(vpaddq_s32(low, high));
        }

        [[clang::always_inline]]
        static i32 ReduceI32(const Vector value) { return vaddvq_s32(AsI32(value)); }

        [[clang::always_inline]]
        static i64 ReduceI64(const Vector value) { return vaddlvq_s32(AsI32(value)); }

        [[clang::always_inline]]
        static Vector PackUnsignedI16(const Vector left, const Vector right)
        { return vreinterpretq_s16_u8(vqmovun_high_s16(vqmovun_s16(left), right)); }

        [[clang::always_inline]]
        static Vector DotU8I8(const Vector sum, const Vector u8, const Vector i8)
        {
            const auto a = vreinterpretq_u8_s16(u8);
            const auto b = vreinterpretq_s8_s16(i8);

            const auto low  = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))), vmovl_s8(vget_low_s8(b)));
            const auto high = vmulq_s16(vreinterpretq_s16_u16(vmovl_high_u8(a)        ), vmovl_high_s8(b)        );

            const auto pairs = vqaddq_s16(vuzp1q_s16(low, high), vuzp2q_s16(low, high));

            return AsI16(vpadalq_s16(AsI32(sum), pairs));
        }

        [[clang::always_inline]]
        static VectorF LoadF(const f32* address) { return vld1q_f32(address); }

        [[clang::always_inline]]
        static VectorF ZeroF() { return vdupq_n_f32(0.0F); }

        [[clang::always_inline]]
        static VectorF AddF(const VectorF left, const VectorF right) { return vaddq_f32(left, right); }

        [[clang::always_inline]]
        static VectorF MulF(const VectorF left, const VectorF right) { return vmulq_f32(left, right); }

        [[clang::always_inline]]
        static f32 ReduceF(const VectorF value)
        { return vpadds_f32(vadd_f32(vget_low_f32(value), vget_high_f32(value))); }

    };

}

#endif

#endif //MANTARAY_BACKEND_ISA_NEON_H
