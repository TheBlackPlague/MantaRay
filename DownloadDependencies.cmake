include(DownloadCPM.cmake)

function(AddGoogleTest)
    CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            VERSION 1.16.0
            OPTIONS
            "INSTALL_GTEST OFF"
            "BUILD_GMOCK OFF"
    )
endfunction()

function(AddGoogleBenchmark)
    CPMAddPackage(
            NAME benchmark
            GITHUB_REPOSITORY google/benchmark
            GIT_TAG 559b7cc1aec1950a9e3f4e879b08cf0b00f796f0
            OPTIONS
            "BENCHMARK_ENABLE_TESTING OFF"
    )
endfunction()