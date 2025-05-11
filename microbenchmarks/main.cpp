//
// Copyright (c) 2025 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#include <cstdlib>

#include <benchmark/benchmark.h>

#include <MantaRay/Backend/Kernel/Activation/ClippedReLU.h>
#include <MantaRay/Frontend/Architecture/Perspective.h>

namespace BM = benchmark;

constexpr auto ClippedReLU = &MantaRay::ClippedReLU<MantaRay::i16, 0, 255>::Activate;

using Starshard = MantaRay::Perspective<MantaRay::i16, MantaRay::i32, ClippedReLU, 768, 256, 1, 512, 400, 255, 64>;
using Aurora    = MantaRay::Perspective<MantaRay::i16, MantaRay::i32, ClippedReLU, 768, 384, 1, 512, 400, 255, 64>;

Starshard StarshardNN;
Aurora       AuroraNN;

// ReSharper disable CppDFAUnreadVariable

void BM_AccumulatorStack_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Push(); StarshardNN.Pop();
}

void BM_AccumulatorStack_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Push(); AuroraNN.Pop();
}

void BM_Refresh_Starshard(BM::State& state)
{
    for (auto _ : state)StarshardNN.Refresh();
}

void BM_Refresh_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Refresh();
}

void BM_Insert_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Insert(0, 0, 8);
}

void BM_Insert_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Insert(0, 0, 8);
}

void BM_Remove_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Remove(0, 0, 8);
}

void BM_Remove_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Remove(0, 0, 8);
}

void BM_Move_Starshard(BM::State& state)
{
    for (auto _ : state) StarshardNN.Move(0, 0, 8, 24);
}

void BM_Move_Aurora(BM::State& state)
{
    for (auto _ : state) AuroraNN.Move(0, 0, 8, 24);
}

void BM_Evaluate_Starshard(BM::State& state)
{
    for (auto _ : state) BM::DoNotOptimize(StarshardNN.Evaluate(0));
}

void BM_Evaluate_Aurora(BM::State& state)
{
    for (auto _ : state) BM::DoNotOptimize(AuroraNN.Evaluate(0));
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
