//
// Copyright (c) 2023 Cerebrum authors. See the list of authors for more details.
// Licensed under MIT.
//

#include "NNUE.h"
#include "ActivationFunction.h"

#include <iostream>
#include <chrono>

// Benchmarking helper class to evaluate performance of Cerebrum.

using PerspectiveNetworkClippedReLU = Cerebrum::PerspectiveNetwork<
        int16_t, Cerebrum::ClippedReLU<int16_t, 0, 255>, 768, 256, 1, 512, 400, 255, 64>;

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


int main()
{
    network.RefreshAccumulator();
    BenchmarkEvaluate(1000000);
}
