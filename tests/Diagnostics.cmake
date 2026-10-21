set(Flags -std=c++23 -fsyntax-only -fno-color-diagnostics -I "${INCLUDE}")

if(COMPILER_TARGET)
    list(APPEND Flags "--target=${COMPILER_TARGET}")
endif()

if(CROSS_SYSROOT)
    list(APPEND Flags "--sysroot=${CROSS_SYSROOT}")
endif()

if(IS_ABSOLUTE "${APPLE_SYSROOT}")
    list(APPEND Flags -isysroot "${APPLE_SYSROOT}")
endif()

execute_process(
        COMMAND "${COMPILER}" ${Flags} "${CASE}"
        RESULT_VARIABLE Result
        OUTPUT_VARIABLE Output
        ERROR_VARIABLE  Output
)

string(FIND "${Output}" "${EXPECTED}" Position)

if(Result EQUAL 0)
    message(FATAL_ERROR "${CASE} compiled, expected '${EXPECTED}'")
elseif(Position EQUAL -1)
    message(FATAL_ERROR "${CASE} did not report '${EXPECTED}':\n${Output}")
endif()
