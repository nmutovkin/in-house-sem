find_package(HDF5 REQUIRED COMPONENTS C)

if(INSEM_REAL STREQUAL "double")
  set(INSEM_REAL_IS_DOUBLE 1)
else()
  set(INSEM_REAL_IS_DOUBLE 0)
endif()
configure_file(include/insem/core/build_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/insem/core/build_config.hpp @ONLY)

add_library(insem STATIC
  src/basis/quadrature.cpp
  src/basis/lagrange.cpp
  src/mesh/macro_mesh.cpp
  src/mesh/forest.cpp
  src/geometry/geometry.cpp
  src/ops/dof_map.cpp
  src/ops/element_block.cpp
  src/cpu/kernels.cpp
  src/solver/locate.cpp
  src/solver/solver.cpp
  src/sim/description.cpp
  src/io/hdf5.cpp
  src/sim/simulation.cpp
)
target_include_directories(insem PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_compile_features(insem PUBLIC cxx_std_17)
set_target_properties(insem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(insem PUBLIC HDF5::HDF5)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(insem PRIVATE -Wall -Wextra -Wpedantic)
  if(INSEM_STRICT_FP)
    target_compile_options(insem PUBLIC -ffp-contract=off)
  endif()
endif()

if(INSEM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(insem PUBLIC OpenMP::OpenMP_CXX)
endif()

if(INSEM_BUILD_APPS)
  add_executable(insem-run apps/insem_run.cpp)
  target_link_libraries(insem-run PRIVATE insem)
  install(TARGETS insem-run RUNTIME DESTINATION bin)
endif()
