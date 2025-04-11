cmake_minimum_required(VERSION 3.15)
project(MantaRay)

option(BUILD_NATIVE "Enable native CPU architecture optimizations" ON)

set(CMAKE_CXX_STANDARD 23)

if(CMAKE_CXX_FLAGS STREQUAL "")
    set(CMAKE_CXX_FLAGS "$ENV{CMAKE_CXX_FLAGS}")
endif()

if(BUILD_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -ftime-report")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_library(MantaRay INTERFACE)
file(COPY src/ DESTINATION include/MantaRay/)
target_include_directories(MantaRay INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")

add_executable(MantaRayBM src/BenchmarkRunner.cpp)
target_link_libraries(MantaRayBM MantaRay)
