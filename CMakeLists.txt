cmake_minimum_required(VERSION 3.25)
project(Cerebrum)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx512f -mavx512bw")

add_library(Cerebrum
        src/SIMD.h src/AccumulatorOperation.h src/Perspective/PerspectiveNNUE.h
        src/Perspective/PerspectiveAccumulator.h src/Activation/ClippedReLU.h src/Backend/Avx.h
        src/Backend/Avx2.h src/Backend/Avx512.h src/Backend/RegisterDefinition.h
        src/IO/DataStream.h src/IO/BinaryStream.h src/IO/MarlinflowStream.h
        src/BenchmarkRunner.cpp src/External/json.hpp )
