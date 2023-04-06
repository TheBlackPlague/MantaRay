cmake_minimum_required(VERSION 3.25)
project(Cerebrum)

set(CMAKE_CXX_STANDARD 20)

add_library(Cerebrum Accumulator.h NNUE.h SIMDOperation.h ActivationFunction.h SIMDBackend.h BenchmarkRunner.cpp)
