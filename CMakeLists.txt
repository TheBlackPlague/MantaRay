cmake_minimum_required(VERSION 3.25)
project(MantaRay)

set(CMAKE_CXX_STANDARD 20)

add_library(MantaRay INTERFACE)
file(COPY src/ DESTINATION include/MantaRay/)
target_include_directories(MantaRay INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")

set_target_properties(MantaRay PROPERTIES LINKER_LANGUAGE CXX)
