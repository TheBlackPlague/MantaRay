//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_BACKEND_KERNEL_GTESTDOT_H
#define MANTARAY_BACKEND_KERNEL_GTESTDOT_H

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <MantaRay/Backend/Lowering.h>
#include <MantaRay/Backend/Target.h>
#include <MantaRay/Backend/Kernel/Update.h>

#include "../ContainerUtil.h"
#include "../../Architecture/Examples.h"

using namespace MantaRay;

namespace DotTest
{

    template<typename A>
    class Fixture
    {

        using P = Backend::Program<A, Backend::Native>;
        using D = Backend::Stage::LayerAt<A, 2>;
        using W = D::WeightSpec;

        constexpr static bool Squared = std::is_same_v<Backend::Stage::LayerAt<A, 0>, SquaredClippedReLU<0, 1>>;

        static_assert(P::Fusable && std::is_same_v<typename W::Type, i16>);
        static_assert(std::is_same_v<typename D::Schedule, std::conditional_t<Squared, DivideBy<255>, Exact>>);

        std::unique_ptr<typename P::Parameters> Parameters = std::make_unique<typename P::Parameters>();

        public:
        constexpr static s00 H       = A::Transformer::OutputSize;
        constexpr static s00 Buckets = D::OutputSize;

        constexpr static Interval Range       = A::Signals[0].Range;
        constexpr static Interval WeightRange = W::Bounds;
        constexpr static Interval BiasRange   = D::BiasSpec::Bounds;

        std::array<std::array<i16, 2 * H>, Buckets> Weights {};
        std::array<i16                   , Buckets> Biases  {};

        void Pack()
        {
            std::vector<i16> weights(Buckets * 2 * H);

            for (s00 o = 0; o < Buckets; o++)
            for (s00 i = 0; i < 2 * H  ; i++) {
                const s00 index = Backend::Stage::BulletIndex<W>(2 * H, Buckets, i, o);

                weights[index] = Weights[o][i];
            }

            P::template StageAt<0>::Pack(weights, Biases, Parameters->template Get<0>());
        }

        i32 Expected(const std::array<i16, H>& stm, const std::array<i16, H>& ntm, const s00 bucket) const
        {
            const auto activate = [](const i16 value) -> i64 {
                const i64 x = std::clamp<i64>(value, 0, 255);

                return Squared ? x * x : x;
            };

            i64 sum = 0;

            for (s00 j = 0; j < H; j++) {
                sum += activate(stm[j]) * Weights[bucket][    j];
                sum += activate(ntm[j]) * Weights[bucket][H + j];
            }

            const i64 value = (Squared ? sum / 255 : sum) + Biases[bucket];

            return static_cast<i32>(value * 400 / 16320);
        }

        i32 Actual(const std::array<i16, H>& stm, const std::array<i16, H>& ntm, const s00 bucket) const
        {
            return P::Forward(*Parameters, stm, ntm, bucket);
        }

        template<s00 Subs, s00 Adds>
        i32 Fused(const Backend::Kernel::Transition<i16, H, Subs, Adds>& transition, const s00 bucket) const
        {
            return P::template ForwardFused<Subs, Adds>(*Parameters, transition, bucket);
        }

    };

    template<typename A>
    void CheckPattern()
    {
        constexpr s00 H = Fixture<A>::H;

        Fixture<A> fixture;

        for (s00 o = 0; o < fixture.Buckets; o++) {
            fixture.Biases[o] = static_cast<i16>(4000 + o);

            for (s00 i = 0; i < 2 * H; i++) fixture.Weights[o][i] = static_cast<i16>((i + o) % 11 - 5);
        }

        fixture.Pack();

        Guarded<i16, H> first ;
        Guarded<i16, H> second;

        for (s00 j = 0; j < H; j++) {
            first [j] = static_cast<i16>(      static_cast<i32>(j % 400) - 50);
            second[j] = static_cast<i16>(300 - static_cast<i32>(j % 400)     );
        }

        for (s00 bucket = 0; bucket < fixture.Buckets; bucket++) {
            const i32 forward  = fixture.Expected(first .Values, second.Values, bucket);
            const i32 backward = fixture.Expected(second.Values, first .Values, bucket);

            EXPECT_EQ(fixture.Actual(first .Values, second.Values, bucket), forward ) << "bucket " << bucket;
            EXPECT_EQ(fixture.Actual(second.Values, first .Values, bucket), backward) << "bucket " << bucket;
        }
    }

    template<typename A>
    void CheckExtremes()
    {
        using F = Fixture<A>;

        constexpr s00 H = F::H;

        F fixture;

        Guarded<i16, H> first ;
        Guarded<i16, H> second;

        for (s00 mode = 0; mode < 4; mode++) {
            const auto weight = [&](const s00 o, const s00 i) {
                if (mode == 2) return (i + o) % 2 == 0 ? F::WeightRange.Max : F::WeightRange.Min;

                return mode == 1 ? F::WeightRange.Min : F::WeightRange.Max;
            };

            for (s00 o = 0; o < F::Buckets; o++) {
                fixture.Biases[o] = static_cast<i16>(mode == 1 ? F::BiasRange.Min : F::BiasRange.Max);

                for (s00 i = 0; i < 2 * H; i++) fixture.Weights[o][i] = static_cast<i16>(weight(o, i));
            }

            fixture.Pack();

            for (s00 j = 0; j < H; j++) {
                first [j] = static_cast<i16>(mode == 3 || (mode == 2 && j % 3 == 0) ? F::Range.Min : F::Range.Max);
                second[j] = static_cast<i16>(mode == 3 || (mode == 2 && j % 5 == 0) ? F::Range.Min : F::Range.Max);
            }

            for (s00 bucket = 0; bucket < F::Buckets; bucket++) {
                const i32 expected = fixture.Expected(first.Values, second.Values, bucket);

                EXPECT_EQ(fixture.Actual(first.Values, second.Values, bucket), expected) << mode << " / " << bucket;
            }
        }
    }

    template<typename A, s00 Subs, s00 Adds>
    void CheckFused(const Fixture<A>& fixture, const u08 stm, const s00 bucket)
    {
        constexpr s00 H = Fixture<A>::H;

        std::array<Guarded<i16, H>, 2> children;
        Guarded<i16, 2, H>             parents ;
        Guarded<i16, 2, H>             outputs ;

        std::array<std::array<Guarded<i16, H>, 2>, 2> removed;
        std::array<std::array<Guarded<i16, H>, 2>, 2> added  ;

        std::array<std::array<const std::array<i16, H>*, Subs>, 2> subtract;
        std::array<std::array<const std::array<i16, H>*, Adds>, 2> add     ;

        for (u08 perspective = 0; perspective < 2; perspective++) {
            for (s00 j = 0; j < H; j++) {
                children[perspective][j] = static_cast<i16>(static_cast<i32>((j * 7 + perspective * 3) % 400) - 60);

                for (s00 k = 0; k < 2; k++) {
                    removed[perspective][k][j] = static_cast<i16>((j + k + perspective) % 13 * 17 - 100);
                    added  [perspective][k][j] = static_cast<i16>((j + 2 * k + perspective) % 19 * 11 - 90);
                }
            }

            parents[perspective] = children[perspective].Values;

            for (s00 k = 0; k < Subs; k++) {
                for (s00 j = 0; j < H; j++)
                    parents[perspective][j] = static_cast<i16>(parents[perspective][j] + removed[perspective][k][j]);

                subtract[perspective][k] = &removed[perspective][k].Values;
            }

            for (s00 k = 0; k < Adds; k++) {
                for (s00 j = 0; j < H; j++)
                    parents[perspective][j] = static_cast<i16>(parents[perspective][j] - added[perspective][k][j]);

                add[perspective][k] = &added[perspective][k].Values;
            }
        }

        const Backend::Kernel::Transition<i16, H, Subs, Adds> transition {
            .Sources      = parents.Values,
            .Destinations = outputs.Values,
            .Subtract     = subtract      ,
            .Add          = add           ,
            .SideToMove   = stm
        };

        const i32 expected = fixture.Expected(children[stm].Values, children[stm ^ 1].Values, bucket);
        const i32 actual   = fixture.template Fused<Subs, Adds>(transition, bucket);

        EXPECT_EQ(actual, expected) << Subs << " / " << Adds;

        for (u08 perspective = 0; perspective < 2; perspective++)
            EXPECT_EQ(outputs[perspective], children[perspective].Values) << Subs << " / " << Adds;

        EXPECT_TRUE(outputs.Intact());
    }

    template<typename A>
    void CheckFusedShapes()
    {
        constexpr s00 H = Fixture<A>::H;

        Fixture<A> fixture;

        for (s00 o = 0; o < fixture.Buckets; o++) {
            fixture.Biases[o] = static_cast<i16>(-3000 + o);

            for (s00 i = 0; i < 2 * H; i++) fixture.Weights[o][i] = static_cast<i16>((3 * i + o) % 23 - 11);
        }

        fixture.Pack();

        for (u08 stm    = 0; stm    < 2              ; stm   ++)
        for (s00 bucket = 0; bucket < fixture.Buckets; bucket++) {
            CheckFused<A, 1, 1>(fixture, stm, bucket);
            CheckFused<A, 2, 1>(fixture, stm, bucket);
            CheckFused<A, 2, 2>(fixture, stm, bucket);
        }
    }

}

TEST(Dot, CReLU_256              ) { DotTest::CheckPattern<Examples::Starshard     >(); }
TEST(Dot, CReLU_384              ) { DotTest::CheckPattern<Examples::Aurora        >(); }
TEST(Dot, CReLU_768              ) { DotTest::CheckPattern<Examples::Flare         >(); }
TEST(Dot, SCReLU_64              ) { DotTest::CheckPattern<Examples::SimpleSCReLU  >(); }
TEST(Dot, SCReLU_768             ) { DotTest::CheckPattern<Examples::WideSCReLU    >(); }
TEST(Dot, SCReLU_Buckets         ) { DotTest::CheckPattern<Examples::OutputBucketed>(); }

TEST(Dot, CReLU_768_Extremes     ) { DotTest::CheckExtremes<Examples::Flare         >(); }
TEST(Dot, SCReLU_64_Extremes     ) { DotTest::CheckExtremes<Examples::SimpleSCReLU  >(); }
TEST(Dot, SCReLU_768_Extremes    ) { DotTest::CheckExtremes<Examples::WideSCReLU    >(); }
TEST(Dot, SCReLU_Buckets_Extremes) { DotTest::CheckExtremes<Examples::OutputBucketed>(); }

TEST(Dot, CReLU_768_Fused        ) { DotTest::CheckFusedShapes<Examples::Flare         >(); }
TEST(Dot, SCReLU_64_Fused        ) { DotTest::CheckFusedShapes<Examples::SimpleSCReLU  >(); }
TEST(Dot, SCReLU_Buckets_Fused   ) { DotTest::CheckFusedShapes<Examples::OutputBucketed>(); }

#endif //MANTARAY_BACKEND_KERNEL_GTESTDOT_H
