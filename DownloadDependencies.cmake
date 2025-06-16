include(DownloadCPM.cmake)

function(AddGoogleHighway)
    CPMAddPackage(
            NAME highway
            GITHUB_REPOSITORY google/highway
            GIT_TAG 8f678418bd0dbf22f155350a1bb085d8af7357ed
            OPTIONS
            "HWY_ENABLE_TESTS OFF"
            "HWY_ENABLE_EXAMPLES OFF"
            "HWY_ENABLE_CONTRIB OFF"
    )
endfunction()

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