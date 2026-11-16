string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ProcessorArchitectureDefinition)

# === Instruction Set ===

set(MantaRayExampleArch "")

if(MANTARAY_ARCH)
    message(STATUS "Examples: MANTARAY_ARCH ${MANTARAY_ARCH}")
elseif(CMAKE_CROSSCOMPILING)
    message(STATUS "Examples: cross-compiling without MANTARAY_ARCH, the compiler's default target")
elseif(ProcessorArchitectureDefinition MATCHES "x86_64|amd64")
    set(MantaRayExampleArch "-march=native")
elseif(ProcessorArchitectureDefinition MATCHES "aarch64|arm64")
    set(MantaRayExampleArch "-mcpu=native")
endif()

# === Examples ===

set(MantaRayExamples
        Perspective
        SquaredPerspective
        KingBuckets
        OutputBuckets
        MultiLayer
        Threats
        PawnPairs
        Combined
        Heads
)

foreach(Example IN LISTS MantaRayExamples)
    set(TARGET_NAME "MantaRayExample-${Example}")
    add_executable(${TARGET_NAME} examples/${Example}.cpp examples/Example.h examples/Position.h)

    target_compile_options(${TARGET_NAME} PRIVATE
            ${MantaRayExampleArch}
            "$<$<CONFIG:Debug>:-Wall;-Wextra>"
    )

    target_link_libraries(${TARGET_NAME} PRIVATE MantaRay)
endforeach()

message(STATUS "Added Examples: ${MantaRayExamples}")
