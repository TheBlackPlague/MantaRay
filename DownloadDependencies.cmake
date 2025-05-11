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
            VERSION 1.9.3
            OPTIONS
            "BENCHMARK_ENABLE_TESTING OFF"
    )
endfunction()