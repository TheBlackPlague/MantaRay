cmake_minimum_required(VERSION 3.21)

project(MantaRay VERSION 4.0.0 LANGUAGES CXX)

# === OPTIONS ===
option(BUILD_TESTS      "Build kernel tests"    ${PROJECT_IS_TOP_LEVEL})
option(BUILD_BENCHMARKS "Build microbenchmarks" ${PROJECT_IS_TOP_LEVEL})

set(MANTARAY_ARCH "" CACHE STRING "Optional architecture compiler flag, e.g. -mavx2")

option(MANTARAY_FORCE_SCALAR "Use portable kernels" OFF)

# === LANGUAGE SETTINGS ===
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === MantaRay Library ===
add_library(MantaRay INTERFACE)
add_library(MantaRay::MantaRay ALIAS MantaRay)

target_compile_features(MantaRay INTERFACE cxx_std_23)

if(MANTARAY_ARCH)
    target_compile_options(MantaRay INTERFACE ${MANTARAY_ARCH})
endif()

if(MANTARAY_FORCE_SCALAR)
    target_compile_definitions(MantaRay INTERFACE MANTARAY_FORCE_SCALAR)
endif()

target_include_directories(MantaRay INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTS OR BUILD_BENCHMARKS)
    include(DownloadDependencies.cmake)
endif()

if(BUILD_TESTS)
    enable_testing()
    AddGoogleTest()
    include(Tests.cmake)
endif()

if(BUILD_BENCHMARKS)
    AddGoogleBenchmark()
    include(MicroBenchmarks.cmake)
endif()
