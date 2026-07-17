cmake_minimum_required(VERSION 3.24)
project(HFT_Arbitrage_Engine VERSION 1.0.0 LANGUAGES CXX)

# --- 1. Global Standards ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Strict ISO C++

# Enable Interprocedural Optimization (LTO) if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)

# --- 2. HFT Compiler Flags ---
if(MSVC)
    # Windows/MSVC Setup
    add_compile_options(
        /W4 /WX               # Strict warnings as errors
        /O2 /Oi /Ot           # Max optimization, inline intrinsics, favor fast code
        /fp:fast              # Fast floating-point math (safe for our arbitrage thresholds)
        /Zc:preprocessor      # Strict standards-conforming preprocessor
        /wd4324               # Suppress struct padding warnings (handled manually in code)
    )
else()
    # Linux/GCC/Clang Setup (Industry Standard for HFT)
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -O3 -march=native -mtune=native # Target local CPU architecture perfectly
        -ffast-math           # Aggressive math optimization
        -fno-rtti             # Zero-overhead: Disable Run-Time Type Information
        -Wno-deprecated-literal-operator # Suppress simdjson warnings in C++20
        -Wno-error=character-conversion  # Suppress GTest warning
        -Wno-error=nan-infinity-disabled # Suppress GTest fast-math warning
    )
endif()

# --- 3. CUDA Device Optimizations ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_LANGUAGE_PROVIDER OR CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3,-v")
    add_compile_definitions(HAS_CUDA)
    message(STATUS "CUDA toolchain detected and activated.")
else()
    message(WARNING "No CUDA compiler found. GPU modules will be skipped.")
endif()

# --- 4. Dependencies ---
include(FetchContent)

# 4.1 GTest Integration
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 4.2 simdjson
FetchContent_Declare(
    simdjson
    URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.10.1.tar.gz
    URL_HASH SHA256=1E8F881CB2C0F626C56CD3665832F1E97B9D4FFC648AD9E1067C134862BBA060
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(simdjson)

if(MSVC)
    set_target_properties(simdjson PROPERTIES SYSTEM ON)
    target_compile_options(simdjson PRIVATE /W0)
endif()

# 4.3 Boost (using find_package for cross-platform, fallback to FetchContent if not found)
# On Mac we can use brew install boost. On Windows we might have a hardcoded path.
find_package(Boost 1.83 QUIET CONFIG)
if(NOT Boost_FOUND)
    # Generic find module if CONFIG fails
    find_package(Boost 1.83 QUIET)
endif()

# 4.4 OpenSSL
find_package(OpenSSL QUIET)

# --- 5. Targets ---

# 5.1 Core Library (Headers and shared logic)
set(CORE_LIB_SOURCES
    src/thread_pinning.cpp
    src/execution_engine.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND CORE_LIB_SOURCES
        src/cuda_test.cu
        src/graph_generator.cu
        src/bellman_ford.cu
        src/live_arbitrage.cu
        src/arbitrage_optimized.cu
    )
endif()

add_library(core_lib STATIC ${CORE_LIB_SOURCES})
target_include_directories(core_lib PUBLIC include)
if(ipo_supported)
    set_target_properties(core_lib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
if(Boost_FOUND)
    target_link_libraries(core_lib PUBLIC Boost::headers Boost::disable_autolinking)
endif()
target_link_libraries(core_lib PUBLIC simdjson::simdjson)

if(OPENSSL_FOUND)
    target_link_libraries(core_lib PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()


# 5.2 E2E Engine Executable
add_executable(hft_engine src/main.cpp)
target_link_libraries(hft_engine PRIVATE core_lib)
if(ipo_supported)
    set_target_properties(hft_engine PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# 5.3 Unit Tests
set(TEST_SOURCES
    tests/main_test.cpp
    tests/dsl_lexer_test.cpp
    tests/dsl_parser_test.cpp
    tests/dsl_compiler_test.cpp
    tests/dsl_vm_test.cpp
    tests/algorithm_test.cpp
    tests/spsc_ringbuffer_test.cpp
    tests/stream_parser_test.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND TEST_SOURCES
        tests/vram_updater_test.cpp
        tests/live_arbitrage_test.cpp
        tests/graph_generator_test.cu
        tests/cross_validation_test.cu
        tests/bf_kernel_v2_test.cu
        tests/numerical_stability_test.cu
        tests/memory_pattern_test.cu
    )
endif()

# Only create unit_tests if there are actually test files.
# But there are.
add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE core_lib GTest::gtest_main)

