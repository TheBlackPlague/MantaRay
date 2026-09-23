# === ARCHITECTURE DETECTION ===

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ProcessorArchitectureDefinition)
set(AMD64 FALSE)
set(ARM64 FALSE)

if(ProcessorArchitectureDefinition MATCHES "x86_64|amd64")
    set(AMD64 TRUE)
elseif(ProcessorArchitectureDefinition MATCHES "aarch64|arm64")
    set(ARM64 TRUE)
endif()

# === Micro Benchmark ===

function(AddMantaRayMicroBench Architecture Extension CompilerFlag)
    set(TARGET_NAME "MantaRayMB-${Architecture}-${Extension}")
    add_executable(${TARGET_NAME} microbenchmarks/main.cpp)

    target_compile_options(${TARGET_NAME} PRIVATE
            ${CompilerFlag}
            "$<$<CONFIG:Debug>:-Wall;-Wextra;-ftime-report>"
    )

    target_link_libraries(${TARGET_NAME} PRIVATE benchmark MantaRay)

    message(STATUS "Added Micro Benchmark for: ${Extension}")
endfunction()

if(AMD64)
    message(STATUS "Platform: AMD64")
    AddMantaRayMicroBench("x86-64" "sse2"     "-msse2"       )
    AddMantaRayMicroBench("x86-64" "sse41"    "-msse4.1"     )

    AddMantaRayMicroBench("x86-64" "avx"      "-mavx"        )
    AddMantaRayMicroBench("x86-64" "avx2"     "-mavx2"       )

    AddMantaRayMicroBench("x86-64" "avx512f"  "-mavx512f"    )
    AddMantaRayMicroBench("x86-64" "avx512bw" "-mavx512bw"   )

    AddMantaRayMicroBench("x86-64" "native"   "-march=native")
elseif(ARM64)
    AddMantaRayMicroBench("arm64" "native" "-mcpu=native")
else()
    AddMantaRayMicroBench("portable" "scalar" "")
endif()
