cmake_minimum_required(VERSION 3.24)

project(insem VERSION 0.1.0 LANGUAGES CXX)

option(INSEM_ENABLE_OPENMP "Build the CPU backend with OpenMP" ON)
option(INSEM_ENABLE_CUDA "Build the CUDA backend (not implemented yet)" OFF)
option(INSEM_ENABLE_MPI "Build with MPI (not implemented yet)" OFF)
option(INSEM_STRICT_FP "Disable FMA contraction for bitwise CPU/CUDA reproducibility" OFF)
option(INSEM_BUILD_TESTS "Build the C++ unit tests" ON)
option(INSEM_BUILD_APPS "Build the insem-run executable" ON)
option(INSEM_BUILD_PYTHON "Build the Python extension module" OFF)
set(INSEM_REAL "float" CACHE STRING "Floating point type of solver fields: float or double")
set_property(CACHE INSEM_REAL PROPERTY STRINGS float double)

if(NOT INSEM_REAL STREQUAL "float" AND NOT INSEM_REAL STREQUAL "double")
  message(FATAL_ERROR "INSEM_REAL must be 'float' or 'double', got '${INSEM_REAL}'")
endif()
if(INSEM_ENABLE_CUDA)
  message(FATAL_ERROR "INSEM_ENABLE_CUDA: the CUDA backend is planned for milestone M5")
endif()
if(INSEM_ENABLE_MPI)
  message(FATAL_ERROR "INSEM_ENABLE_MPI: MPI support is planned for milestone M8")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(cpp)

if(INSEM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(INSEM_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/src/bindings.cpp)
  target_link_libraries(_core PRIVATE insem)
  install(TARGETS _core LIBRARY DESTINATION insem)
endif()
