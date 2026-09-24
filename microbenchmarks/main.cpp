//
// Copyright (c) 2026 MantaRay authors. See the list of authors for more details.
// Licensed under MIT.
//

#include <benchmark/benchmark.h>

#include <MantaRay/MantaRay.h>

namespace BM = benchmark;

using namespace MantaRay;

template<s00 Hidden>
using Architecture = Network<
    Quantization<255, 64, 400>,
    Mirror<
        Accumulate<
            Layer<768, Hidden, ClippedReLU<0, 1>>
        >
    >,
    Concat,
    Layer<2 * Hidden, 1>
>;

using Starshard = Runtime::Network<Architecture<256>>;
using    Aurora = Runtime::Network<Architecture<384>>;

using StarshardStack = Runtime::AccumulatorStack<Architecture<256>, 512>;
using    AuroraStack = Runtime::AccumulatorStack<Architecture<384>, 512>;

Starshard StarshardNN;
Aurora       AuroraNN;

StarshardStack StarshardAccumulatorStack;
   AuroraStack    AuroraAccumulatorStack;

// ReSharper disable CppDFAUnreadVariable

void BM_AccumulatorStack_Starshard(BM::State& state)
{
    for (auto _ : state) {
        StarshardAccumulatorStack++;

        BM::DoNotOptimize(&*StarshardAccumulatorStack);
        BM::ClobberMemory();

        StarshardAccumulatorStack--;
    }
}

void BM_AccumulatorStack_Aurora(BM::State& state)
{
    for (auto _ : state) {
        AuroraAccumulatorStack++;

        BM::DoNotOptimize(&*AuroraAccumulatorStack);
        BM::ClobberMemory();

        AuroraAccumulatorStack--;
    }
}

void BM_Refresh_Starshard(BM::State& state)
{
    for (auto _ : state) {
        StarshardNN.Refresh(*StarshardAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Refresh_Aurora(BM::State& state)
{
    for (auto _ : state) {
        AuroraNN.Refresh(*AuroraAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Insert_Starshard(BM::State& state)
{
    for (auto _ : state) {
        StarshardNN.Insert(0, 0, 8, *StarshardAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Insert_Aurora(BM::State& state)
{
    for (auto _ : state) {
        AuroraNN.Insert(0, 0, 8, *AuroraAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Remove_Starshard(BM::State& state)
{
    for (auto _ : state) {
        StarshardNN.Remove(0, 0, 8, *StarshardAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Remove_Aurora(BM::State& state)
{
    for (auto _ : state) {
        AuroraNN.Remove(0, 0, 8, *AuroraAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Move_Starshard(BM::State& state)
{
    for (auto _ : state) {
        StarshardNN.Move(0, 0, 8, 24, *StarshardAccumulatorStack);

        BM::ClobberMemory();
    }
}

void BM_Move_Aurora(BM::State& state)
{
    for (auto _ : state) {
        AuroraNN.Move(0, 0, 8, 24, *AuroraAccumulatorStack);

        BM::ClobberMemory();
    }
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
