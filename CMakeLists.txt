cmake_minimum_required(VERSION 3.15)
project(MantaRay)

option(BUILD_NATIVE "Enable native CPU architecture optimizations" ON)
option(BUILD_TEST "Enable tests" ON)

set(CMAKE_CXX_STANDARD 23)

if(CMAKE_CXX_FLAGS STREQUAL "")
    set(CMAKE_CXX_FLAGS "$ENV{CMAKE_CXX_FLAGS}")
endif()

if(BUILD_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -ftime-report")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CPM_DOWNLOAD_VERSION 0.40.8)

if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    message(STATUS "Setting CPM.cmake location to ${CPM_DOWNLOAD_LOCATION}")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    message(STATUS "Setting CPM.cmake location to ${CPM_DOWNLOAD_LOCATION}")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    message(STATUS "Setting CPM.cmake location to ${CPM_DOWNLOAD_LOCATION}")
endif()

set(CPM_DOWNLOAD_RETRY_COUNT 3)
foreach(DOWNLOAD_ATTEMPT RANGE 1 ${CPM_DOWNLOAD_RETRY_COUNT})
    if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
        message(STATUS "CPM.cmake was not found at ${CPM_DOWNLOAD_LOCATION}")
        message(STATUS "Attempt: ${DOWNLOAD_ATTEMPT}/${CPM_DOWNLOAD_RETRY_COUNT}: Downloading CPM.cmake...")
        file(DOWNLOAD
                https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
                ${CPM_DOWNLOAD_LOCATION}
                STATUS DOWNLOAD_STAT
        )
        list(GET DOWNLOAD_STAT 0 DOWNLOAD_RES)
        if(NOT DOWNLOAD_RES EQUAL 0)
            if (DOWNLOAD_ATTEMPT EQUAL CPM_DOWNLOAD_RETRY_COUNT)
                message(FATAL_ERROR "Max download attempts exceeded.")
                break()
            endif()

            message(STATUS "Download failed. Removing ${CPM_DOWNLOAD_LOCATION} and retrying in 5 seconds...")
            file(REMOVE ${CPM_DOWNLOAD_LOCATION})
            execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 5)
        endif()
    endif()

    if(EXISTS ${CPM_DOWNLOAD_LOCATION})
        message(STATUS "CPM.cmake was found at ${CPM_DOWNLOAD_LOCATION}")
        break()
    endif()
endforeach()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.16.0
        OPTIONS
            "INSTALL_GTEST OFF"
            "BUILD_GMOCK OFF"
)

add_library(MantaRay INTERFACE)
file(COPY src/ DESTINATION include/MantaRay/)
target_include_directories(MantaRay INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")

add_executable(MantaRayBM src/BenchmarkRunner.cpp)
target_link_libraries(MantaRayBM MantaRay)

file(GLOB MantaRayBackend
        "src2/Backend/Kernel/Activation/*.h"
        "src2/Backend/Kernel/*.h"
        "src2/Backend/SIMD/*.h"
        "src2/Backend/*.h"
)

add_library(MantaRayV2 INTERFACE)
file(COPY src2/ DESTINATION include/MantaRayV2/)
target_include_directories(MantaRayV2 INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")

if(BUILD_TEST)
    file(GLOB MantaRayBackendTest
            "tests/Backend/Kernel/*.h"
            "tests/Backend/*.h"
    )

    add_executable(MantaRayTest tests/main.cpp
            ${MantaRayBackendTest}
    )
    target_link_libraries(MantaRayTest gtest gtest_main MantaRayV2)

    add_test(NAME MantaRay COMMAND MantaRayTest)
endif()

add_executable(MantaRayV2Assembler src2/main.cpp
        ${MantaRayBackend}
)
