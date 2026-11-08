//
// Copyright (c) 2026 Shaheryar Sohail
// SPDX-License-Identifier: MIT
//

#ifndef MANTARAY_ARCHITECTURE_GTESTANALYSIS_H
#define MANTARAY_ARCHITECTURE_GTESTANALYSIS_H

#include <array>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <MantaRay/Architecture/Activation/ReLU.h>
#include <MantaRay/Architecture/Activation/SquaredReLU.h>
#include <MantaRay/Backend/ISA/Scalar.h>
#include <MantaRay/Backend/Lowering.h>
#include <MantaRay/Backend/Target.h>
#include <MantaRay/Runtime/Network.h>

#include "Examples.h"

using namespace MantaRay;

namespace AnalysisTest
{

    constexpr i64 Giga = 1000000000;

    template<Signal In>
    constexpr Scale Ratio = Divide(Scale { 400 * In.Scale.Denominator, 1 }, In.Scale.Numerator);

    template<Signal In>
    constexpr bool Narrows = Fits<i32>(In.Range * Interval { Ratio<In>.Numerator, Ratio<In>.Numerator });

    template<Signal In>
    consteval bool MatchesRuntimeScale(const i64 step)
    {
        for (i64 value = In.Range.Min; value < In.Range.Max; value += step)
            if (Output<400>::Apply<In>(value) != Output<400>::Apply(value, In.Scale)) return false;

        return Output<400>::Apply<In>(In.Range.Max) == Output<400>::Apply(In.Range.Max, In.Scale);
    }

    template<typename ISA, typename A>
    constexpr std::string_view Describe = Backend::Program<A, ISA>::Describe();

    template<typename ISA>
    consteval bool PinLowering()
    {
        constexpr bool Scalar = std::is_same_v<ISA, Backend::ISA::Scalar>;

        static_assert(ISA::DotChains == (Scalar ? 4 : 2));

        constexpr std::string_view CReLU = Scalar    ? "PerspectiveDot<CReLU,K=4,i32>"           :
                                           ISA::VNNI ? "PerspectiveDot<CReLU-PackusDot,K=2,i32>" :
                                                       "PerspectiveDot<CReLU,K=2,i32>"           ;

        constexpr std::string_view SCReLU  = Scalar ? "PerspectiveDot<SCReLU-Lizard,K=4,i32>"
                                                    : "PerspectiveDot<SCReLU-Lizard,K=2,i32>";
        constexpr std::string_view Wide    = Scalar ? "PerspectiveDot<SCReLU-Lizard,K=4,i64>"
                                                    : "PerspectiveDot<SCReLU-Lizard,K=2,i64>";
        constexpr std::string_view Buckets = Scalar ? "PerspectiveDot<SCReLU-Lizard,K=4,i32,Select=8>"
                                                    : "PerspectiveDot<SCReLU-Lizard,K=2,i32,Select=8>";

        static_assert(Describe<ISA, Examples::Starshard     > == CReLU  );
        static_assert(Describe<ISA, Examples::Aurora        > == CReLU  );
        static_assert(Describe<ISA, Examples::Flare         > == CReLU  );
        static_assert(Describe<ISA, Examples::SimpleSCReLU  > == SCReLU );
        static_assert(Describe<ISA, Examples::WideSCReLU    > == Wide   );
        static_assert(Describe<ISA, Examples::OutputBucketed> == Buckets);
        static_assert(Describe<ISA, Examples::KingBucketed  > == Buckets);

        constexpr std::string_view SquareFirst = Scalar ? "PerspectiveDot<SCReLU-SquareFirst,K=4,i32>"
                                                        : "PerspectiveDot<SCReLU-SquareFirst,K=2,i32>";
        constexpr std::string_view Linear      = Scalar ? "PerspectiveDot<ReLU,K=4,i32>"
                                                        : "PerspectiveDot<ReLU,K=2,i32>";

        static_assert(Describe<ISA, Examples::SquareFirstSCReLU> == SquareFirst);
        static_assert(Describe<ISA, Examples::LinearReLU       > == Linear     );

        static_assert(Describe<ISA, Examples::WidenedSCReLU >.starts_with("PerspectiveDot<SCReLU-Widened,K=" ));
        static_assert(Describe<ISA, Examples::WidenedSqrReLU>.starts_with("PerspectiveDot<SqrReLU-Widened,K="));
        static_assert(Describe<ISA, Examples::WidenedSCReLU >.ends_with(",i64>"));
        static_assert(Describe<ISA, Examples::WidenedSqrReLU>.ends_with(",i64>"));

        static_assert(Describe<ISA, Examples::MultiLayer> ==
                      "PairwiseConcat<ZeroPoint> -> DenseAffine<ZeroPoint,Dequantize,Select=8,SCReLU> -> "
                      "DenseAffine<Float,Select=8,SCReLU> -> DenseAffine<Float,Select=8,Output>");
        static_assert(Describe<ISA, Examples::PackedPairwise> ==
                      "PairwiseConcat<U8> -> DenseAffine<U8,Dequantize,SCReLU> -> DenseAffine<Float,Output>");
        static_assert(Describe<ISA, Examples::ShiftedPairwise> == "PairwiseConcat<I16> -> DenseAffine<I16,Output>");
        static_assert(Describe<ISA, Examples::WideAccumulator> ==
                      "Single<SquaredClippedReLU> -> Single<Concat> -> Single<Affine> -> Single<Output>");

        return true;
    }

}

TEST(Analysis, Flare)
{
    using A = Examples::Flare;

    static_assert(A::LayerCount == 4 && A::Signals.size() == 5);

    static_assert(A::Signals[0].Range        == Interval { -16665, 16665 });
    static_assert(A::Signals[0].Scale        == Scale    { 255, 1 });
    static_assert(A::Signals[0].Size         == 768);
    static_assert(A::Signals[0].Perspectives == 2);

    static_assert(A::Signals[1].Range == Interval { 0, 255 });

    static_assert(A::Signals[2].Size == 1536 && A::Signals[2].Perspectives == 1);

    static_assert(A::Signals[3].Range == Repeat(Interval { 0, 255 } * Interval { -127, 127 }, 1536) +
                                         Interval { -32768, 32767 });
    static_assert(A::Signals[3].Scale == Scale { 16320, 1 });

    static_assert(A::Signals[4].Range == Divide(A::Signals[3].Range * Interval { 400, 400 }, 16320));
    static_assert(A::Signals[4].Size  == 1);
}

TEST(Analysis, SimpleSCReLU)
{
    using A = Examples::SimpleSCReLU;

    static_assert(A::Signals[1].Scale == Scale    { 65025, 1 });
    static_assert(A::Signals[1].Range == Interval { 0, 65025 });

    static_assert(A::Signals[3].Scale == Scale { 16320, 1 });
    static_assert(A::Signals[3].Range == Divide(Repeat(Interval { 0, 65025 } * Interval { -127, 127 }, 128), 255) +
                                         Interval { -32768, 32767 });
}

TEST(Analysis, MultiLayer)
{
    using A = Examples::MultiLayer;

    static_assert(A::LayerCount == 11);

    static_assert(A::Signals[1].Range        == Interval { 0, 65025 });
    static_assert(A::Signals[1].Scale        == Scale    { 65025, 1 });
    static_assert(A::Signals[1].Size         == 32);
    static_assert(A::Signals[1].Perspectives == 2);

    static_assert(A::Signals[3].Domain  == Domain::Float);
    static_assert(A::Signals[3].Size    == 128);
    static_assert(A::Signals[3].Buckets == 8);

    static_assert(A::Signals[4].Size    == 16);
    static_assert(A::Signals[4].Buckets == 1);

    static_assert(A::Signals[5].Domain == Domain::Float);
    static_assert(A::Signals[7].Size   == 32);

    static_assert(A::Signals.back().Size   == 1);
    static_assert(A::Signals.back().Domain == Domain::Integer);
}

TEST(Analysis, KingBucketed)
{
    using A = Examples::KingBucketed;
    using F = A::Transformer;

    static_assert(F::Features::MaxActive == 32);
    static_assert(F::WeightSpec::Bounds  == Interval { -505, 505 });
    static_assert(F::BiasSpec  ::Bounds  == Interval { -505, 505 });

    static_assert(A::Signals[0].Range == Repeat(Interval { -505, 505 }, 32) + Interval { -505, 505 });
    static_assert(A::Signals[0].Range == Interval { -16665, 16665 });

    static_assert(A::Signals[3].Size    == 8);
    static_assert(A::Signals[3].Buckets == 8);
    static_assert(A::Signals[3].Scale   == Scale { 16320, 1 });

    static_assert(A::Signals[4].Size    == 1);
    static_assert(A::Signals[4].Buckets == 1);
}

TEST(Analysis, PairwiseShift)
{
    using A = Architecture<
        FeatureTransformer<Chess768, 64, Weights<i16, 255, Examples::Bound>, Biases<i16, 255, Examples::Bound>>,
        Pairwise<ClippedReLU<0, 1>, 9>,
        Concat,
        Affine<64, 1, Weights<i08, 64, Examples::Bound>, Biases<f32>, Dequantize>,
        Output<400>
    >;

    static_assert(A::Signals[1].Range == Interval { 0, 127 });
    static_assert(A::Signals[1].Scale == Scale    { 65025, 512 });
    static_assert(A::Signals[1].Size  == 32);

    static_assert(A::Signals[3].Domain == Domain::Float);
}

TEST(Analysis, ShiftRightSchedule)
{
    using A = Examples::WideAccumulator;

    static_assert(A::Signals[0].Range == Repeat(Interval { -32768, 32767 }, 32) + Interval { -32768, 32767 });
    static_assert(A::Signals[1].Scale == Scale { 65536, 1 });
    static_assert(A::Signals[3].Scale == Scale { 16384, 1 });
    static_assert(A::Signals[3].Range == ShiftDown(Repeat(Interval { 0, 65536 } * Interval { -127, 127 }, 64), 8) +
                                         Interval { std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max() });
}

TEST(Analysis, ReLUFamilies)
{
    using A = Architecture<
        FeatureTransformer<Chess768, 32, Weights<i16, 255, Examples::Bound>, Biases<i16, 255, Examples::Bound>>,
        ReLU,
        SquaredReLU,
        Concat,
        Affine<64, 1, Weights<i16, 64, Examples::Bound>, Biases<f32>, Dequantize>,
        Output<400>
    >;

    static_assert(A::Signals[1].Range == Interval { 0, 16665 });
    static_assert(A::Signals[1].Scale == Scale    { 255, 1 });
    static_assert(A::Signals[2].Range == Interval { 0, 16665 * 16665 });
    static_assert(A::Signals[2].Scale == Scale    { 65025, 1 });
}

TEST(Analysis, ConceptsAndTypes)
{
    using A = Examples::Perspective<64>;

    static_assert(std::is_same_v<A::Layers, List<ClippedReLU<0, 1>, Concat,
        Affine<128, 1, Weights<i16, 64, Examples::Bound>, Biases<i16, 255 * 64>>, Output<400>>>);
    static_assert(A::Transformer::OutputSize == 64);

    static_assert(Layer<ReLU> && Layer<ClippedReLU<>> && Layer<SquaredClippedReLU<>> && Layer<SquaredReLU>);
    static_assert(Layer<Pairwise<ClippedReLU<>>> && Layer<Concat> && Layer<Select<MaterialCount<8>>>);
    static_assert(Layer<Affine<64, 1, Weights<f32>, Biases<f32>>> && Layer<Output<400>>);
}

TEST(Tensor, Bounds)
{
    using Feature = Weights<i16, 255, Clip<-1.98, 1.98>>;
    using Dense   = Weights<i16,  64, Clip<-1.98, 1.98>, Transposed>;
    using Narrow  = Weights<i08, 255, Clip<-1.98, 1.98>>;
    using Bias    = Biases <i16, 255 * 64>;
    using Real    = Weights<f32>;

    static_assert(std::is_same_v<Feature::Type, i16> && Feature::Quantization == 255);
    static_assert(Feature::IsClipped && !Feature::IsTransposed);
    static_assert(Feature::Bounds == Interval { -505, 505 });

    static_assert(Dense::IsClipped && Dense::IsTransposed);
    static_assert(Dense::Bounds == Interval { -127, 127 });

    static_assert(Narrow::Bounds == Interval { -128, 127 });

    static_assert(!Bias::IsClipped && Bias::Quantization == 16320);
    static_assert(Bias::Bounds == Interval { -32768, 32767 });

    static_assert(std::is_same_v<Real::Type, f32> && Real::Quantization == 1);
    static_assert(Real::Bounds == Interval { 0, 0 });

    static_assert(Weights<i16, 1, Clip<-0.5 , 0.5 >>::Bounds == Interval { -1, 1 });
    static_assert(Weights<i16, 2, Clip<-0.25, 0.25>>::Bounds == Interval { -1, 1 });
    static_assert(Weights<i16, 2, Clip<-0.24, 0.24>>::Bounds == Interval {  0, 0 });
}

TEST(Activation, Reference)
{
    constexpr Scale Q { 255, 1 };

    static_assert(ClippedReLU< 0, 1>::Apply( 300, Q) ==  255);
    static_assert(ClippedReLU< 0, 1>::Apply(  -7, Q) ==    0);
    static_assert(ClippedReLU< 0, 1>::Apply( 100, Q) ==  100);
    static_assert(ClippedReLU<-1, 2>::Apply(-600, Q) == -255);
    static_assert(ClippedReLU<-1, 2>::Apply( 600, Q) ==  510);

    static_assert(SquaredClippedReLU<0, 1>::Apply(300, Q) == 65025);
    static_assert(SquaredClippedReLU<0, 1>::Apply( -3, Q) ==     0);
    static_assert(SquaredClippedReLU<0, 1>::Apply( 10, Q) ==   100);

    static_assert(ReLU::Apply(-5, Q) == 0);
    static_assert(ReLU::Apply( 7, Q) == 7);

    static_assert(SquaredReLU::Apply( -5, Q) ==     0);
    static_assert(SquaredReLU::Apply(300, Q) == 90000);

    static_assert(ClippedReLU<0, 1>::ApplyFloat( 1.50F) == 1.00F);
    static_assert(ClippedReLU<0, 1>::ApplyFloat(-0.50F) == 0.00F);
    static_assert(ClippedReLU<0, 1>::ApplyFloat( 0.25F) == 0.25F);

    static_assert(SquaredClippedReLU<0, 1>::ApplyFloat(0.5F) == 0.25F);
    static_assert(SquaredClippedReLU<0, 1>::ApplyFloat(3.0F) == 1.00F);

    static_assert(ReLU       ::ApplyFloat(-2.0F) == 0.0F);
    static_assert(SquaredReLU::ApplyFloat( 3.0F) == 9.0F);
    static_assert(SquaredReLU::ApplyFloat(-3.0F) == 0.0F);
}

TEST(Schedule, Reference)
{
    static_assert(Exact        ::Apply(-511) == -511);
    static_assert(DivideBy<255>::Apply(-511) ==   -2);
    static_assert(DivideBy<255>::Apply( 511) ==    2);
    static_assert(ShiftRight<8>::Apply(-257) ==   -2);
    static_assert(ShiftRight<8>::Apply( 257) ==    1);

    static_assert(DivideBy<255>::Apply(Scale { 4161600, 1 }) == Scale { 16320,   1 });
    static_assert(ShiftRight<9>::Apply(Scale {   65025, 1 }) == Scale { 65025, 512 });

    static_assert(DivideBy<2>  ::Apply(Interval { -5, 9 }) == Interval { -2, 4 });
    static_assert(ShiftRight<1>::Apply(Interval { -5, 9 }) == Interval { -3, 4 });
}

TEST(Output, CompileTimeSignal)
{
    using AnalysisTest::Giga;
    using AnalysisTest::MatchesRuntimeScale;
    using AnalysisTest::Narrows;
    using AnalysisTest::Ratio;

    constexpr Signal Narrow { .Size = 1, .Perspectives = 1, .Range = { -16500000, 16500000 }, .Scale = { 16320, 1 } };
    constexpr Signal Wide   { .Size = 1, .Perspectives = 1, .Range = { -5 * Giga, 5 * Giga }, .Scale = { 16320, 1 } };
    constexpr Signal Shift  { .Size = 1, .Perspectives = 1, .Range = {  -1000000,  1000000 }, .Scale = { 65025, 512 } };

    constexpr i64 Limit = std::numeric_limits<i32>::max() / Ratio<Narrow>.Numerator;

    constexpr Signal Edge { .Size = 1, .Perspectives = 1, .Range = { -Limit    , Limit     }, .Scale = { 16320, 1 } };
    constexpr Signal Over { .Size = 1, .Perspectives = 1, .Range = { -Limit - 1, Limit + 1 }, .Scale = { 16320, 1 } };

    static_assert(Ratio<Narrow> == Scale { 5, 204 } && Ratio<Shift> == Scale { 8192, 2601 });
    static_assert(Edge.Range.Max == 429496729 && Over.Range.Max == 429496730);

    static_assert( Narrows<Narrow> &&  Narrows<Edge>);
    static_assert(!Narrows<Over  > && !Narrows<Wide> && !Narrows<Shift>);

    static_assert(MatchesRuntimeScale<Narrow>(   65537));
    static_assert(MatchesRuntimeScale<Edge  >(39916801));
    static_assert(MatchesRuntimeScale<Over  >(39916801));
    static_assert(MatchesRuntimeScale<Wide  >(39916801));
    static_assert(MatchesRuntimeScale<Shift >(    7919));

    static_assert(Output<400>::Apply<Narrow>(     -40) ==         0);
    static_assert(Output<400>::Apply<Narrow>(     -41) ==        -1);
    static_assert(Output<400>::Apply<Narrow>(      41) ==         1);
    static_assert(Output<400>::Apply<Narrow>(16500000) ==    404411);
    static_assert(Output<400>::Apply<Wide  >(5 * Giga) == 122549019);
}

TEST(Output, Reference)
{
    static_assert(Output<400>::Apply( 1000000, Scale { 16320,   1 }) ==  24509);
    static_assert(Output<400>::Apply(-1000000, Scale { 16320,   1 }) == -24509);
    static_assert(Output<400>::Apply(   65025, Scale { 65025, 512 }) == 204800);

    static_assert(Output<400>::ApplyFloat( 1.2345F) ==  493);
    static_assert(Output<400>::ApplyFloat(-1.2345F) == -493);
}

TEST(Output, FloatSaturates)
{
    constexpr f32 Inf = std::numeric_limits<f32>::infinity();

    constexpr i32 Max = std::numeric_limits<i32>::max();
    constexpr i32 Min = std::numeric_limits<i32>::min();

    static_assert(Output<400>::ApplyFloat( 1.0E30F) == Max);
    static_assert(Output<400>::ApplyFloat(-1.0E30F) == Min);
    static_assert(Output<400>::ApplyFloat( Inf    ) == Max);
    static_assert(Output<400>::ApplyFloat(-Inf    ) == Min);

    static_assert(Output<1>::ApplyFloat( 2147483648.0F) == Max);
    static_assert(Output<1>::ApplyFloat(-2147483648.0F) == Min);
    static_assert(Output<1>::ApplyFloat( 2147483520.0F) ==  2147483520);
    static_assert(Output<1>::ApplyFloat(-2147483520.0F) == -2147483520);

    static_assert(Output<1>::ApplyFloat( 2.75F) ==  2);
    static_assert(Output<1>::ApplyFloat(-2.75F) == -2);
    static_assert(Output<1>::ApplyFloat(-0.50F) ==  0);

    static_assert(Output<400>::ApplyFloat( 0.0024F) == 0);
    static_assert(Output<400>::ApplyFloat(-0.0024F) == 0);

    const f32 NaN = std::numeric_limits<f32>::quiet_NaN();

    EXPECT_EQ(Output<400>::ApplyFloat( NaN), 0);
    EXPECT_EQ(Output<400>::ApplyFloat(-NaN), 0);
    EXPECT_EQ(Output<1  >::ApplyFloat( NaN), 0);
}

TEST(Lowering, Describe)
{
    static_assert(AnalysisTest::PinLowering<Backend::ISA::Scalar>());
    static_assert(AnalysisTest::PinLowering<Backend::Native     >());

    static_assert(Network<Examples::Flare>::Describe() == AnalysisTest::Describe<Backend::Native, Examples::Flare>);

    static_assert(Network<Examples::Flare>::Fusable && !Network<Examples::MultiLayer>::Fusable);
}

TEST(Lowering, ProgramShape)
{
    using Flare = Backend::Program<Examples::Flare          , Backend::ISA::Scalar>;
    using Multi = Backend::Program<Examples::MultiLayer     , Backend::ISA::Scalar>;
    using Wide  = Backend::Program<Examples::WideAccumulator, Backend::ISA::Scalar>;

    static_assert(Flare::Stages::Size == 1 && Multi::Stages::Size == 4 && Wide::Stages::Size == 4);

    static_assert(std::is_same_v<Flare::Accumulator, i16>);
    static_assert(std::is_same_v<Multi::Accumulator, i16>);
    static_assert(std::is_same_v<Wide ::Accumulator, i32>);

    using Dot = Backend::Stage::PerspectiveDot<Examples::Flare, Backend::ISA::Scalar, 0, 4>;

    static_assert(std::is_same_v<Flare::Stages, List<Dot>>);
    static_assert(Dot::Plan.Mode == Backend::Kernel::DotMode::Linear);
    static_assert(Dot::Plan.Accumulators == 4 && !Dot::Plan.WideReduce);
    static_assert(Dot::Plan.Low == 0 && Dot::Plan.High == 255);

    static_assert(alignof(Flare::Parameters) == Alignment && alignof(Multi::Parameters) == Alignment);
    static_assert(std::is_same_v<decltype(Dot::Parameters::Weights), Array<i16, 1, 2, 768>>);
    static_assert(std::is_same_v<decltype(std::declval<Flare::Parameters&>().Get<0>()), Dot::Parameters&>);

#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
    using Packus = Backend::Stage::PerspectiveDot<Examples::Flare, Backend::ISA::AVX512VNNI, 0, 4>;

    static_assert(Packus::Plan.Mode == Backend::Kernel::DotMode::PackusDot);
    static_assert(Packus::Plan.Accumulators == 2 && !Packus::Plan.WideReduce);
    static_assert(std::is_same_v<decltype(Packus::Parameters::Weights), Array<i08, 1, 2, 768>>);
#endif
}

TEST(Lowering, Target)
{
#if defined(MANTARAY_FORCE_SCALAR)
    static_assert(std::is_same_v<Backend::Native, Backend::ISA::Scalar>);
#elif defined(MANTARAY_EXPECTED_ISA)
    static_assert(std::is_same_v<Backend::Native, Backend::ISA::MANTARAY_EXPECTED_ISA>);
#else
    GTEST_SKIP() << "the native target is not pinned";
#endif
}

#endif //MANTARAY_ARCHITECTURE_GTESTANALYSIS_H
