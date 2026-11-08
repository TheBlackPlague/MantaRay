string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ProcessorArchitectureDefinition)
set(AMD64 FALSE)
set(ARM64 FALSE)

if(ProcessorArchitectureDefinition MATCHES "x86_64|amd64")
    set(AMD64 TRUE)
elseif(ProcessorArchitectureDefinition MATCHES "aarch64|arm64")
    set(ARM64 TRUE)
endif()

# === Tests ===

file(GLOB_RECURSE MantaRayTests CONFIGURE_DEPENDS tests/*.h)

function(AddMantaRayTest Architecture Extension CompilerFlag)
    set(TARGET_NAME "MantaRayTest-${Architecture}-${Extension}")
    add_executable(${TARGET_NAME} tests/main.cpp ${MantaRayTests})

    target_compile_options(${TARGET_NAME} PRIVATE
            ${CompilerFlag}
            "$<$<CONFIG:Debug>:-Wall;-Wextra;-ftime-report>"
    )

    if(ARGN)
        target_compile_definitions(${TARGET_NAME} PRIVATE "MANTARAY_EXPECTED_ISA=${ARGN}")
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE gtest gtest_main MantaRay)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})

    message(STATUS "Added Test for: ${Extension}")
endfunction()

function(AddMantaRayDiagnostic Case Expected)
    add_test(NAME MantaRayDiagnostic-${Case} COMMAND ${CMAKE_COMMAND}
            "-DCOMPILER=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_TARGET=${CMAKE_CXX_COMPILER_TARGET}"
            "-DCROSS_SYSROOT=${CMAKE_SYSROOT}"
            "-DAPPLE_SYSROOT=${CMAKE_OSX_SYSROOT}"
            "-DINCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DCASE=${CMAKE_CURRENT_SOURCE_DIR}/tests/Architecture/Diagnostics/${Case}.cpp"
            "-DEXPECTED=${Expected}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/Diagnostics.cmake"
    )
endfunction()

if(AMD64)
    message(STATUS "Platform: AMD64")
    AddMantaRayTest("x86-64" "sse2"       "-msse2"                  SSE2      )
    AddMantaRayTest("x86-64" "sse41"      "-msse4.1"                SSE41     )

    AddMantaRayTest("x86-64" "avx"        "-mavx"                   SSE41     )
    AddMantaRayTest("x86-64" "avx2"       "-mavx2"                  AVX2      )

    AddMantaRayTest("x86-64" "avx512f"    "-mavx512f"               AVX2      )
    AddMantaRayTest("x86-64" "avx512bw"   "-mavx512bw"              AVX512    )
    AddMantaRayTest("x86-64" "avx512vnni" "-mavx512bw;-mavx512vnni" AVX512VNNI)

    AddMantaRayTest("x86-64" "native"     "-march=native"                     )
elseif(ARM64)
    message(STATUS "Platform: ARM64")

    if(CMAKE_CROSSCOMPILING)
        AddMantaRayTest("arm64" "native" "-mcpu=cortex-a76" NEON)
    else()
        AddMantaRayTest("arm64" "native" "-mcpu=native"     NEON)
    endif()
else()
    AddMantaRayTest("portable" "scalar" "" Scalar)
endif()

AddMantaRayDiagnostic(ConcatTwice        "layer 3 (Concat): Concat requires two perspectives")
AddMantaRayDiagnostic(LastNotOutput      "layer 3 (Affine): the last layer must be Output")
AddMantaRayDiagnostic(MissingRescale     "sum scale 4161600 does not match bias scale 16320; use DivideBy<255>")
AddMantaRayDiagnostic(NonSimdHidden      "layer 0 (FeatureTransformer): hidden size 760 is not a multiple of 32")
AddMantaRayDiagnostic(OutputMidChain     "layer 3 (Output): Output must be the last layer")
AddMantaRayDiagnostic(OutputNotLast      "layer 4 (Output): Output must be the last layer")
AddMantaRayDiagnostic(PairwiseShift      "Pairwise requires a shift below 63")
AddMantaRayDiagnostic(SelectNotDivisible "layer 4 (Select): size 7 is not divisible into 8 buckets")
AddMantaRayDiagnostic(WrongAffineInput   "layer 3 (Affine): expected input size 1535 but previous layer produces 1536")
