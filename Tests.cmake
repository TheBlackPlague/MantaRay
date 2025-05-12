# === ARCHITECTURE DETECTION ===

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

    target_link_libraries(${TARGET_NAME} PRIVATE gtest gtest_main MantaRay)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})

    message(STATUS "Added Test for: ${Extension}")
endfunction()

if(AMD64)
    message(STATUS "Platform: AMD64")
    AddMantaRayTest("x86-64" "sse2"     "-msse2"       )
    AddMantaRayTest("x86-64" "sse41"    "-msse4.1"     )

    AddMantaRayTest("x86-64" "avx"      "-mavx"        )
    AddMantaRayTest("x86-64" "avx2"     "-mavx2"       )

    AddMantaRayTest("x86-64" "avx512f"  "-mavx512f"    )
    AddMantaRayTest("x86-64" "avx512bw" "-mavx512bw"   )

    AddMantaRayTest("x86-64" "native"   "-march=native")
elseif(ARM64)
    if(APPLE)
        message(STATUS "Platform: Apple Silicon")
        AddMantaRayTest("arm64" "m1" "-mcpu=apple-m1")
        AddMantaRayTest("arm64" "m2" "-mcpu=apple-m2")
        AddMantaRayTest("arm64" "m3" "-mcpu=apple-m3")
        AddMantaRayTest("arm64" "m4" "-mcpu=apple-m4")
    else()
        message(STATUS "Platform: ARM64")
        AddMantaRayTest("arm64" "neon" "-mfpu=neon")
    endif()
endif()