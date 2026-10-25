#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "insem/mesh/macro_mesh.hpp"

namespace insem::test {

constexpr double kPi = 3.14159265358979323846;

// nx x nz grid of Quad4 cells on [x0, x1] x [z0, z1], with boundary tags
// 1 = bottom, 2 = right, 3 = top, 4 = left. `shear` skews the grid into
// parallelograms (x += shear * z).
inline std::shared_ptr<const mesh::MacroMesh> grid(int nx, int nz, double x0 = 0.0,
                                                   double x1 = 1.0, double z0 = 0.0,
                                                   double z1 = 1.0, double shear = 0.0,
                                                   mesh::CoordSystem coords = mesh::CoordSystem::Cartesian) {
  std::vector<double> nodes;
  for (int j = 0; j <= nz; ++j) {
    for (int i = 0; i <= nx; ++i) {
      const double z = z0 + (z1 - z0) * j / nz;
      nodes.push_back(x0 + (x1 - x0) * i / nx + shear * z);
      nodes.push_back(z);
    }
  }
  auto node = [nx](int i, int j) { return static_cast<index_t>(i + (nx + 1) * j); };
  std::vector<index_t> cells;
  std::vector<int> material;
  std::vector<mesh::BoundaryFace> boundary;
  for (int j = 0; j < nz; ++j) {
    for (int i = 0; i < nx; ++i) {
      const auto c = static_cast<index_t>(material.size());
      cells.insert(cells.end(), {node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)});
      material.push_back(0);
      if (j == 0) boundary.push_back({c, 0, 1});
      if (i == nx - 1) boundary.push_back({c, 1, 2});
      if (j == nz - 1) boundary.push_back({c, 2, 3});
      if (i == 0) boundary.push_back({c, 3, 4});
    }
  }
  return std::make_shared<const mesh::MacroMesh>(mesh::CellType::Quad4, coords, std::move(nodes),
                                                 std::move(cells), std::move(material),
                                                 std::move(boundary));
}

// Quarter annulus r in [r0, r1], angle in [0, pi/2], as nr x nt Quad9 cells
// with exact node positions on circles (curved geometry).
inline std::shared_ptr<const mesh::MacroMesh> quarter_annulus(int nr, int nt, double r0 = 1.0, double r1 = 2.0) {
  const int mr = 2 * nr, mt = 2 * nt;  // quadratic node lattice
  std::vector<double> nodes;
  for (int j = 0; j <= mt; ++j) {
    for (int i = 0; i <= mr; ++i) {
      const double r = r0 + (r1 - r0) * i / mr;
      const double t = 0.5 * kPi * j / mt;
      nodes.push_back(r * std::cos(t));
      nodes.push_back(r * std::sin(t));
    }
  }
  auto node = [mr](int i, int j) { return static_cast<index_t>(i + (mr + 1) * j); };
  std::vector<index_t> cells;
  std::vector<int> material;
  for (int j = 0; j < nt; ++j) {
    for (int i = 0; i < nr; ++i) {
      const int a = 2 * i, b = 2 * j;
      cells.insert(cells.end(), {node(a, b), node(a + 2, b), node(a + 2, b + 2), node(a, b + 2),
                                 node(a + 1, b), node(a + 2, b + 1), node(a + 1, b + 2),
                                 node(a, b + 1), node(a + 1, b + 1)});
      material.push_back(0);
    }
  }
  return std::make_shared<const mesh::MacroMesh>(mesh::CellType::Quad9, mesh::CoordSystem::Cartesian,
                                                 std::move(nodes), std::move(cells),
                                                 std::move(material));
}

}  // namespace insem::test
