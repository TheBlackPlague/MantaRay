include(DownloadCPM.cmake)

CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.16.0
        OPTIONS
        "INSTALL_GTEST OFF"
        "BUILD_GMOCK OFF"
)

CPMAddPackage(
        NAME nanobench
        GITHUB_REPOSITORY martinus/nanobench
        VERSION 4.3.11
        GIT_SHALLOW TRUE
)