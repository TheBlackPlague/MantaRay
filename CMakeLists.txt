cmake_minimum_required(VERSION 3.21)
project(MantaRay LANGUAGES CXX)

# === OPTIONS ===
option(BUILD_TEST "Build tests" ON)

# === LANGUAGE SETTINGS ===
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === DEPENDENCIES ===
include(DownloadDependencies.cmake)

# === MantaRay Library ===
add_library(MantaRay INTERFACE)

target_include_directories(MantaRay INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# === MantaRay Tests ===
if(BUILD_TEST)
    message(STATUS "Building tests")
    include(Tests.cmake)
endif()
