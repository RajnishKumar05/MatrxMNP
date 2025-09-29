cmake_minimum_required(VERSION 3.15)
project(cppmatrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP
find_package(OpenMP)

# pybind11 headers
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYBIND11_INCLUDES})

# Conda Python 3.13 headers
include_directories(/root/miniconda3/include/python3.13)

# Build Python module
add_library(cppmatrix MODULE bindings.cpp)
set_target_properties(cppmatrix PROPERTIES PREFIX "" SUFFIX ".so")

# Link OpenBLAS manually
target_link_libraries(cppmatrix PRIVATE /usr/lib/x86_64-linux-gnu/libopenblas.so)

# OpenMP
if(OpenMP_CXX_FOUND)
   target_link_libraries(cppmatrix PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optimization
target_compile_options(cppmatrix PRIVATE -O3 -march=native -funroll-loops)
