cmake_minimum_required(VERSION 3.12)
project(bitinfer)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add C++ sources
set(SOURCES
    cpp/kernels.cpp
    bindings/pybind.cpp
)

# Create pybind11 module
pybind11_add_module(bitinfer_cpp ${SOURCES})

# Compiler-specific optimizations
target_compile_definitions(bitinfer_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Metal framework for macOS
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    target_link_libraries(bitinfer_cpp PRIVATE ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
endif()
