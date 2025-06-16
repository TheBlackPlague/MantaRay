cmake_minimum_required(VERSION 3.21)
project(MantaRay LANGUAGES CXX)

# === OPTIONS ===
option(BUILD_TEST "Build tests" ON)
option(BUILD_MB "Build micro benchmarks" ON)

# === LANGUAGE SETTINGS ===
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === DEPENDENCIES ===
include(DownloadDependencies.cmake)
AddGoogleHighway()

# === MantaRay Library ===
add_library(MantaRay INTERFACE)

target_include_directories(MantaRay INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(MantaRay INTERFACE hwy)

# === MantaRay Tests ===
if(BUILD_TEST)
    AddGoogleTest()
    message(STATUS "Building tests")
    include(Tests.cmake)
endif()

# === MantaRay Micro Benchmarks ===
if(BUILD_MB)
    AddGoogleBenchmark()
    message(STATUS "Building micro benchmarks")
    include(MicroBenchmarks.cmake)
endif()
