//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#include "Perspective/PerspectiveNNUE.h"
#include "Activation/ClippedReLU.h"

#include <iostream>
#include <chrono>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wxor-used-as-pow"

// Benchmarking helper class to evaluate performance of Cerebrum.

using PerspectiveNetworkClippedReLU = Cerebrum::PerspectiveNetwork<
        int16_t, int32_t, Cerebrum::ClippedReLU<int16_t, 0, 255>, 768, 256, 1, 512, 400, 255, 64>;

static PerspectiveNetworkClippedReLU network = PerspectiveNetworkClippedReLU();

void BenchmarkEvaluate(const int samples)
{
    long long timeSum = 0;
    int output;
    for (int i = 0; i < samples; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        output = network.Evaluate(0);
        auto stop = std::chrono::high_resolution_clock::now();
        timeSum += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    }
    auto timeAvg = (double)timeSum / samples;
    std::cout << "Evaluation output was " << output << " and took " << timeAvg << "ns!" << std::endl;
}

void EmulateBoardStartPosition()
{
    // Pawns
    for (int sq = 8; sq < 16; sq++) {
        network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(0, 0, sq);
        network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(0, 1, sq ^ 56);
    }
    // Knights
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(1, 0, 1);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(1, 0, 6);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(1, 1, 1 ^ 56);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(1, 1, 6 ^ 56);

    // Bishops
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(2, 0, 2);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(2, 0, 5);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(2, 1, 2 ^ 56);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(2, 1, 5 ^ 56);

    // Rooks
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(3, 0, 0);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(3, 0, 7);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(3, 1, 0 ^ 56);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(3, 1, 7 ^ 56);

    // Queen
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(4, 0, 3);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(4, 1, 3 ^ 56);

    // King
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(5, 0, 4);
    network.EfficientlyUpdateAccumulator<Cerebrum::AccumulatorOperation::Activate>(5, 1, 4 ^ 56);
}

int main()
{
    network.RefreshAccumulator();
    EmulateBoardStartPosition();
    BenchmarkEvaluate(1000000);
}

#pragma clang diagnostic pop