cmake_minimum_required(VERSION 3.18)

project(
    Proxima
    VERSION 0.1.0
    LANGUAGES CXX
)

option(
    PROXIMA_BUILD_TESTS
    "Build C++ tests"
    ON
)

option(
    PROXIMA_BUILD_BENCHMARKS
    "Build benchmarks"
    ON
)

option(
    PROXIMA_BUILD_PYTHON
    "Build Python bindings"
    OFF
)

option(
    PROXIMA_ENABLE_AVX2
    "Enable AVX2"
    OFF
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    CMAKE_EXPORT_COMPILE_COMMANDS
    ON
)

# --------------------------------------------------
# Library
# --------------------------------------------------

add_library(
    proxima
    src/hnsw.cpp
    src/dist/dispatch.cpp
)

target_include_directories(
    proxima
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(
    proxima
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

# --------------------------------------------------
# Compiler warnings
# --------------------------------------------------

if(MSVC)

    target_compile_options(
        proxima
        PRIVATE
            /W4
            /permissive-
    )

else()

    target_compile_options(
        proxima
        PRIVATE
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
    )

endif()

# --------------------------------------------------
# Optimization / SIMD
# --------------------------------------------------

if(PROXIMA_ENABLE_AVX2)

    if(MSVC)

        target_compile_options(
            proxima
            PRIVATE
                /arch:AVX2
        )

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

        target_compile_options(
            proxima
            PRIVATE
                -mavx2
                -mfma
        )

    endif()

endif()

# --------------------------------------------------
# Tests
# --------------------------------------------------

if(PROXIMA_BUILD_TESTS)

    enable_testing()

    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(
        gtest_force_shared_crt
        ON
        CACHE BOOL ""
        FORCE
    )

    FetchContent_MakeAvailable(googletest)

    add_executable(
        test_proxima
        tests/cpp/test_proxima.cpp
    )

    target_link_libraries(
        test_proxima
        PRIVATE
            proxima
            GTest::gtest_main
    )

    if(
        CMAKE_CXX_COMPILER_ID MATCHES
        "GNU|Clang"
    )

        target_compile_options(
            test_proxima
            PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
        )

        target_link_options(
            test_proxima
            PRIVATE
                -fsanitize=address,undefined
        )

    endif()

    include(GoogleTest)

    gtest_discover_tests(test_proxima)

endif()

# --------------------------------------------------
# C++ Benchmarks
# --------------------------------------------------

if(PROXIMA_BUILD_BENCHMARKS)

    add_executable(
        bench_proxima
        benchmarks/bench_proxima.cpp
    )

    target_link_libraries(
        bench_proxima
        PRIVATE
            proxima
    )

endif()

# --------------------------------------------------
# Python bindings
# --------------------------------------------------

if(PROXIMA_BUILD_PYTHON)

    find_package(
        Python
        COMPONENTS
            Interpreter
            Development.Module
            Development.SABIModule
        REQUIRED
    )

    execute_process(
        COMMAND
            "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE nanobind_result
    )

    if(NOT nanobind_result EQUAL 0)
        message(
            FATAL_ERROR
            "Could not locate nanobind CMake files"
        )
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")

    find_package(
        nanobind
        CONFIG
        REQUIRED
    )

    nanobind_add_module(
        _proxima
        bindings/python.cpp
    )

    target_link_libraries(
        _proxima
        PRIVATE
            proxima
    )

    target_include_directories(
        _proxima
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    install(
        TARGETS
            _proxima
        LIBRARY
            DESTINATION proxima
        RUNTIME
            DESTINATION proxima
        ARCHIVE
            DESTINATION proxima
    )

endif()