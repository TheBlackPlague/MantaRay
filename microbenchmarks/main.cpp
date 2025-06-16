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

void BM_Move_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Normal(0, 0, 8, 24, *StarshardAccumulatorStack);
}

void BM_Move_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Normal(0, 0, 8, 24, *AuroraAccumulatorStack);
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

BENCHMARK(BM_Move_Starshard);
BENCHMARK(BM_Move_Aurora);

BENCHMARK(BM_Evaluate_Starshard);
BENCHMARK(BM_Evaluate_Aurora);

BENCHMARK_MAIN();
