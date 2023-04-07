cmake_minimum_required(VERSION 3.25)
project(Cerebrum)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2")

add_library(Cerebrum Accumulator.h NNUE.h SIMDOperation.h ActivationFunction.h SIMDBackend.h BenchmarkRunner.cpp)
