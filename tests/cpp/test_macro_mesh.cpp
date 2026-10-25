#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "insem/core/error.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using namespace insem;
using namespace insem::mesh;

TEST_CASE("Grid topology: edges, neighbours, boundary", "[mesh]") {
  const auto m = test::grid(3, 2);
  CHECK(m->n_cells() == 6);
  CHECK(m->n_nodes() == 12);
  // nx (nz + 1) horizontal + nz (nx + 1) vertical edges.
  CHECK(m->n_edges() == 3 * 3 + 2 * 4);
  int boundary_edges = 0;
  for (index_t e = 0; e < m->n_edges(); ++e) boundary_edges += m->is_boundary_edge(e) ? 1 : 0;
  CHECK(boundary_edges == 2 * (3 + 2));
  // Cells 0 and 1 share face 1 of cell 0 / face 3 of cell 1.
  CHECK(m->cell_edge(0, 1) == m->cell_edge(1, 3));
  CHECK(m->cell_edge_forward(0, 1) != m->cell_edge_forward(1, 3));
  CHECK(m->boundary().size() == 10);
}

TEST_CASE("Quad4 mapping and Jacobian", "[mesh]") {
  const auto m = test::grid(1, 1, 2.0, 5.0, -1.0, 1.0, 0.5);
  const auto x = m->map(0, 0.0, 0.0);
  CHECK_THAT(x[0], WithinAbs(3.5, 1e-14));
  CHECK_THAT(x[1], WithinAbs(0.0, 1e-14));
  const auto j = m->jacobian(0, 0.3, -0.2);
  CHECK_THAT(j[0], WithinAbs(1.5, 1e-14));  // dx/dxi
  CHECK_THAT(j[1], WithinAbs(0.5, 1e-14));  // dx/deta (shear)
  CHECK_THAT(j[2], WithinAbs(0.0, 1e-14));
  CHECK_THAT(j[3], WithinAbs(1.0, 1e-14));
}

TEST_CASE("Quad9 cells represent circles", "[mesh]") {
  const auto m = test::quarter_annulus(1, 1, 1.0, 2.0);
  // Node positions are reproduced exactly; mid-face points lie on the circle
  // to the accuracy of quadratic interpolation.
  const auto corner = m->map(0, 1.0, 1.0);
  CHECK_THAT(corner[0], WithinAbs(0.0, 1e-14));
  CHECK_THAT(corner[1], WithinAbs(2.0, 1e-14));
  const auto p = m->map(0, 1.0, 0.5);
  CHECK_THAT(std::hypot(p[0], p[1]), WithinAbs(2.0, 2e-2));
}

TEST_CASE("Invalid meshes are rejected", "[mesh]") {
  const std::vector<double> nodes{0, 0, 1, 0, 1, 1, 0, 1};
  // Clockwise cell.
  CHECK_THROWS_AS(MacroMesh(CellType::Quad4, CoordSystem::Cartesian, nodes, {0, 3, 2, 1}, {0}),
                  insem::Error);
  // Out-of-range node.
  CHECK_THROWS_AS(MacroMesh(CellType::Quad4, CoordSystem::Cartesian, nodes, {0, 1, 2, 7}, {0}),
                  insem::Error);
  // Wrong cell array size.
  CHECK_THROWS_AS(MacroMesh(CellType::Quad9, CoordSystem::Cartesian, nodes, {0, 1, 2, 3}, {0}),
                  insem::Error);
  // Negative radius in axisymmetric mode.
  const std::vector<double> shifted{-1, 0, 1, 0, 1, 1, -1, 1};
  CHECK_THROWS_AS(MacroMesh(CellType::Quad4, CoordSystem::Axisymmetric, shifted, {0, 1, 2, 3}, {0}),
                  insem::Error);
  // Boundary tag on an interior face.
  const auto g = test::grid(2, 1);
  CHECK_THROWS_AS(MacroMesh(CellType::Quad4, CoordSystem::Cartesian, g->nodes(), g->cells(),
                            g->materials(), {{0, 1, 7}}),
                  insem::Error);
}

TEST_CASE("Overlapping cells sharing an edge in the same direction are rejected", "[mesh]") {
  // Two cells folded onto each other across edge (1, 2).
  const std::vector<double> nodes{0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.0, 0.5, 1.0};
  CHECK_THROWS_AS(MacroMesh(CellType::Quad4, CoordSystem::Cartesian, nodes,
                            {0, 1, 2, 3, 4, 1, 2, 5}, {0, 0}),
                  insem::Error);
}
