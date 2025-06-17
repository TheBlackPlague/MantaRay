//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#include <cstdlib>

#include <benchmark/benchmark.h>

#include <MantaRay/Frontend/Architecture/Perspective.h>
#include <MantaRay/Frontend/Architecture/Common/AccumulatorStack.h>

namespace BM = benchmark;

using ClippedReLU = MantaRay::ClippedReLU<MantaRay::i16, 0, 255>;

using Starshard = MantaRay::Perspective<MantaRay::i16, MantaRay::i32, ClippedReLU, 768, 256, 1, 400, 255, 64>;
using Aurora    = MantaRay::Perspective<MantaRay::i16, MantaRay::i32, ClippedReLU, 768, 384, 1, 400, 255, 64>;

using StarshardStack = MantaRay::AccumulatorStack<MantaRay::i16, 256, 512>;
using    AuroraStack = MantaRay::AccumulatorStack<MantaRay::i16, 384, 512>;

Starshard StarshardNN;
Aurora       AuroraNN;

StarshardStack StarshardAccumulatorStack;
   AuroraStack    AuroraAccumulatorStack;

// ReSharper disable CppDFAUnreadVariable

void BM_AccumulatorStack_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardAccumulatorStack++; StarshardAccumulatorStack--;
}

void BM_AccumulatorStack_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraAccumulatorStack++; AuroraAccumulatorStack--;
}

void BM_Refresh_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Refresh(*StarshardAccumulatorStack);
}

void BM_Refresh_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Refresh(*AuroraAccumulatorStack);
}

void BM_Insert_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Insert(0, 0, 8, *StarshardAccumulatorStack);
}

void BM_Insert_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Insert(0, 0, 8, *AuroraAccumulatorStack);
}

void BM_Remove_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Remove(0, 0, 8, *StarshardAccumulatorStack);
}

void BM_Remove_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Remove(0, 0, 8, *AuroraAccumulatorStack);
}

void BM_Normal_Starshard(BM::State& state)
{
    constexpr Starshard::AccumulatorUpdateNormal update {
        .Piece  =  0,
        .Side   =  0,
        .Origin = 12,
        .Target = 28
    };

    for (auto _ : state) StarshardNN.DispatchUpdate(update, *StarshardAccumulatorStack);
}

void BM_Normal_Aurora(BM::State& state)
{
    constexpr Aurora::AccumulatorUpdateNormal update {
        .Piece  =  0,
        .Side   =  0,
        .Origin = 12,
        .Target = 28
    };

    for (auto _ : state) AuroraNN.DispatchUpdate(update, *AuroraAccumulatorStack);
}

void BM_Capture_Starshard(BM::State& state)
{
    constexpr Starshard::AccumulatorUpdateCapture update {
        .VictimPiece =  0,
        .Piece       =  0,
        .Side        =  0,
        .Origin      = 28,
        .Target      = 35
    };

    for (auto _ : state) StarshardNN.DispatchUpdate(update, *StarshardAccumulatorStack);
}

void BM_Capture_Aurora(BM::State& state)
{
    constexpr Aurora::AccumulatorUpdateCapture update {
        .VictimPiece =  0,
        .Piece       =  0,
        .Side        =  0,
        .Origin      = 28,
        .Target      = 35
    };

    for (auto _ : state) AuroraNN.DispatchUpdate(update, *AuroraAccumulatorStack);
}

void BM_Promotion_Starshard(BM::State& state)
{
    constexpr Starshard::AccumulatorUpdatePromotion update {
        .PromotionPiece =  4,
        .Piece          =  0,
        .Side           =  0,
        .Origin         = 48,
        .Target         = 56
    };

    for (auto _ : state) StarshardNN.DispatchUpdate(update, *StarshardAccumulatorStack);
}

void BM_Promotion_Aurora(BM::State& state)
{
    constexpr Aurora::AccumulatorUpdatePromotion update {
        .PromotionPiece =  4,
        .Piece          =  0,
        .Side           =  0,
        .Origin         = 48,
        .Target         = 56
    };

    for (auto _ : state) AuroraNN.DispatchUpdate(update, *AuroraAccumulatorStack);
}

void BM_PromotionCapture_Starshard(BM::State& state)
{
    constexpr Starshard::AccumulatorUpdatePromotionCapture update {
        .PromotionPiece =  4,
        .VictimPiece    =  2,
        .Piece          =  0,
        .Side           =  0,
        .Origin         = 48,
        .Target         = 57
    };

    for (auto _ : state) StarshardNN.DispatchUpdate(update, *StarshardAccumulatorStack);
}

void BM_PromotionCapture_Aurora(BM::State& state)
{
    constexpr Aurora::AccumulatorUpdatePromotionCapture update {
        .PromotionPiece =  4,
        .VictimPiece    =  2,
        .Piece          =  0,
        .Side           =  0,
        .Origin         = 48,
        .Target         = 57
    };

    for (auto _ : state) AuroraNN.DispatchUpdate(update, *AuroraAccumulatorStack);
}

void BM_Castle_Starshard(BM::State& state)
{
    constexpr Starshard::AccumulatorUpdateCastle update {
        .Side          = 0,
        .OriginKing    = 4,
        .TargetKing    = 6,
        .OriginRook    = 7,
        .TargetRook    = 5
    };

    for (auto _ : state) StarshardNN.DispatchUpdate(update, *StarshardAccumulatorStack);
}

void BM_Castle_Aurora(BM::State& state)
{
    constexpr Aurora::AccumulatorUpdateCastle update {
        .Side          = 0,
        .OriginKing    = 4,
        .TargetKing    = 6,
        .OriginRook    = 7,
        .TargetRook    = 5
    };

    for (auto _ : state) AuroraNN.DispatchUpdate(update, *AuroraAccumulatorStack);
}

void BM_Evaluate_Starshard(BM::State& state)
{
    for (auto _ : state) BM::DoNotOptimize(
        StarshardNN.Evaluate(0, *StarshardAccumulatorStack)
    );
}

void BM_Evaluate_Aurora(BM::State& state)
{
    for (auto _ : state) BM::DoNotOptimize(
        AuroraNN.Evaluate(0, *AuroraAccumulatorStack)
    );
}

BENCHMARK(BM_AccumulatorStack_Starshard);
BENCHMARK(BM_AccumulatorStack_Aurora);

BENCHMARK(BM_Refresh_Starshard);
BENCHMARK(BM_Refresh_Aurora);

BENCHMARK(BM_Insert_Starshard);
BENCHMARK(BM_Insert_Aurora);

BENCHMARK(BM_Remove_Starshard);
BENCHMARK(BM_Remove_Aurora);

BENCHMARK(BM_Normal_Starshard);
BENCHMARK(BM_Normal_Aurora);

BENCHMARK(BM_Capture_Starshard);
BENCHMARK(BM_Capture_Aurora);

BENCHMARK(BM_Promotion_Starshard);
BENCHMARK(BM_Promotion_Aurora);

BENCHMARK(BM_PromotionCapture_Starshard);
BENCHMARK(BM_PromotionCapture_Aurora);

BENCHMARK(BM_Castle_Starshard);
BENCHMARK(BM_Castle_Aurora);

BENCHMARK(BM_Evaluate_Starshard);
BENCHMARK(BM_Evaluate_Aurora);

BENCHMARK_MAIN();
