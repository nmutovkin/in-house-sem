find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(insem_tests
  test_quadrature.cpp
  test_lagrange.cpp
  test_macro_mesh.cpp
  test_forest.cpp
  test_dof_map.cpp
  test_geometry.cpp
  test_hdf5.cpp
  test_simulation.cpp
)
target_link_libraries(insem_tests PRIVATE insem Catch2::Catch2WithMain)
catch_discover_tests(insem_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
