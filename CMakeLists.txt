cmake_policy(SET CMP0135 NEW)
cmake_minimum_required(VERSION 3.16)
project(Proxima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------
# Compiler warnings
# --------------------------------

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        # -Werror
    )
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    add_compile_options(-mavx2 -mfma)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(-march=armv8-a+simd)
endif()

# ----------------------------
# Library
# ----------------------------

file(GLOB DIST_SOURCES src/dist/*.cpp)

add_library(hnsw
    src/hnsw.cpp
    ${DIST_SOURCES}
)

target_include_directories(hnsw
    PUBLIC ${PROJECT_SOURCE_DIR}/src
)

target_compile_options(hnsw PRIVATE -O3)

# ----------------------------
# Benchmark executable
# ----------------------------

add_executable(bench
    benchmarks/bench.cpp
)

target_link_libraries(bench PRIVATE hnsw)

# ----------------------------
# GoogleTest via FetchContent
# ----------------------------

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_hnsw
    tests/test_hnsw.cpp
)

target_link_libraries(test_hnsw
    PRIVATE
    hnsw
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_hnsw)