#include "insem/mesh/macro_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

#include "insem/core/error.hpp"

namespace insem::mesh {
namespace {

// Reference-lattice position (0, 1, 2 for xi = -1, 0, 1) of the Quad9 nodes.
constexpr int kQuad9A[9] = {0, 2, 2, 0, 1, 2, 1, 0, 1};
constexpr int kQuad9B[9] = {0, 0, 2, 2, 0, 1, 2, 1, 1};
constexpr double kQuad4Xi[4] = {-1.0, 1.0, 1.0, -1.0};
constexpr double kQuad4Eta[4] = {-1.0, -1.0, 1.0, 1.0};

// 1D quadratic Lagrange polynomials on {-1, 0, 1} and their derivatives.
void quadratic(double t, double* l, double* dl) {
  l[0] = 0.5 * t * (t - 1.0);
  l[1] = 1.0 - t * t;
  l[2] = 0.5 * t * (t + 1.0);
  dl[0] = t - 0.5;
  dl[1] = -2.0 * t;
  dl[2] = t + 0.5;
}

// Shape functions and their reference derivatives at (xi, eta).
int shape(CellType type, double xi, double eta, double* n, double* dn_dxi, double* dn_deta) {
  if (type == CellType::Quad4) {
    for (int k = 0; k < 4; ++k) {
      const double a = 1.0 + kQuad4Xi[k] * xi;
      const double b = 1.0 + kQuad4Eta[k] * eta;
      n[k] = 0.25 * a * b;
      dn_dxi[k] = 0.25 * kQuad4Xi[k] * b;
      dn_deta[k] = 0.25 * a * kQuad4Eta[k];
    }
    return 4;
  }
  double lx[3], dlx[3], ly[3], dly[3];
  quadratic(xi, lx, dlx);
  quadratic(eta, ly, dly);
  for (int k = 0; k < 9; ++k) {
    n[k] = lx[kQuad9A[k]] * ly[kQuad9B[k]];
    dn_dxi[k] = dlx[kQuad9A[k]] * ly[kQuad9B[k]];
    dn_deta[k] = lx[kQuad9A[k]] * dly[kQuad9B[k]];
  }
  return 9;
}

}  // namespace

std::string to_string(CellType type) { return type == CellType::Quad4 ? "quad4" : "quad9"; }

std::string to_string(CoordSystem coords) {
  return coords == CoordSystem::Cartesian ? "cartesian" : "axisymmetric";
}

CellType cell_type_from_string(const std::string& name) {
  if (name == "quad4") return CellType::Quad4;
  if (name == "quad9") return CellType::Quad9;
  throw Error("unknown cell type '" + name + "'");
}

CoordSystem coord_system_from_string(const std::string& name) {
  if (name == "cartesian") return CoordSystem::Cartesian;
  if (name == "axisymmetric") return CoordSystem::Axisymmetric;
  throw Error("unknown coordinate system '" + name + "'");
}

MacroMesh::MacroMesh(CellType type, CoordSystem coords, std::vector<double> nodes,
                     std::vector<index_t> cells, std::vector<int> material,
                     std::vector<BoundaryFace> boundary)
    : type_(type),
      coords_(coords),
      nodes_(std::move(nodes)),
      cells_(std::move(cells)),
      material_(std::move(material)),
      boundary_(std::move(boundary)) {
  INSEM_CHECK(nodes_.size() % 2 == 0, "node coordinate array must hold (x, z) pairs");
  INSEM_CHECK(cells_.size() == material_.size() * nodes_per_cell(),
              "cell array has " << cells_.size() << " entries, expected "
                                << material_.size() * nodes_per_cell() << " for "
                                << material_.size() << " " << to_string(type_) << " cells");
  const index_t nn = n_nodes();
  for (index_t v : cells_) {
    INSEM_CHECK(v >= 0 && v < nn, "cell references node " << v << " outside [0, " << nn << ")");
  }
  build_topology();
  validate();
}

void MacroMesh::build_topology() {
  const index_t nc = n_cells();
  cell_edges_.assign(4 * static_cast<std::size_t>(nc), -1);
  cell_edge_forward_.assign(4 * static_cast<std::size_t>(nc), 0);
  edge_nodes_.clear();
  edge_cells_.clear();

  std::unordered_map<std::uint64_t, index_t> lookup;
  lookup.reserve(4 * static_cast<std::size_t>(nc));
  for (index_t c = 0; c < nc; ++c) {
    for (int f = 0; f < 4; ++f) {
      const index_t a = cell_node(c, kFaceCorners[f][0]);
      const index_t b = cell_node(c, kFaceCorners[f][1]);
      INSEM_CHECK(a != b, "cell " << c << " has a degenerate face " << f);
      const index_t lo = std::min(a, b);
      const index_t hi = std::max(a, b);
      const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint32_t>(hi);
      auto [it, inserted] = lookup.try_emplace(key, n_edges());
      const index_t e = it->second;
      if (inserted) {
        edge_nodes_.push_back(lo);
        edge_nodes_.push_back(hi);
        edge_cells_.push_back(c);
        edge_cells_.push_back(-1);
      } else {
        INSEM_CHECK(edge_cells_[2 * e + 1] < 0,
                    "edge (" << lo << ", " << hi << ") is shared by more than two cells");
        // Two counter-clockwise cells traverse a shared edge in opposite
        // directions; the same direction means inverted or overlapping cells.
        const index_t other = edge_cells_[2 * e];
        int other_face = 0;
        while (cell_edges_[4 * other + other_face] != e) ++other_face;
        INSEM_CHECK((a < b) != (cell_edge_forward_[4 * other + other_face] != 0),
                    "cells " << other << " and " << c << " have inconsistent orientation");
        edge_cells_[2 * e + 1] = c;
      }
      cell_edges_[4 * c + f] = e;
      cell_edge_forward_[4 * c + f] = a < b ? 1 : 0;
    }
  }
}

void MacroMesh::validate() const {
  const index_t nc = n_cells();
  // Positive Jacobian on a 3x3 sample of reference points.
  for (index_t c = 0; c < nc; ++c) {
    for (double xi : {-1.0, 0.0, 1.0}) {
      for (double eta : {-1.0, 0.0, 1.0}) {
        const auto j = jacobian(c, xi, eta);
        const double det = j[0] * j[3] - j[1] * j[2];
        INSEM_CHECK(det > 0.0, "cell " << c << " has a non-positive Jacobian (" << det
                                       << ") at (xi, eta) = (" << xi << ", " << eta
                                       << "); cells must be counter-clockwise and non-degenerate");
      }
    }
  }
  if (coords_ == CoordSystem::Axisymmetric) {
    double extent = 0.0;
    for (double v : nodes_) extent = std::max(extent, std::abs(v));
    for (index_t i = 0; i < n_nodes(); ++i) {
      INSEM_CHECK(nodes_[2 * i] >= -1e-12 * extent,
                  "axisymmetric mesh has node " << i << " at negative r = " << nodes_[2 * i]);
    }
  }
  for (const auto& bf : boundary_) {
    INSEM_CHECK(bf.cell >= 0 && bf.cell < nc, "boundary face references cell " << bf.cell);
    INSEM_CHECK(bf.face >= 0 && bf.face < 4, "boundary face index " << bf.face << " not in [0, 4)");
    INSEM_CHECK(is_boundary_edge(cell_edge(bf.cell, bf.face)),
                "face " << bf.face << " of cell " << bf.cell
                        << " is tagged as boundary but is shared with another cell");
  }
}

std::array<double, 2> MacroMesh::map(index_t c, double xi, double eta) const {
  double n[9], dxi[9], deta[9];
  const int nk = shape(type_, xi, eta, n, dxi, deta);
  std::array<double, 2> x{0.0, 0.0};
  for (int k = 0; k < nk; ++k) {
    const index_t v = cell_node(c, k);
    x[0] += n[k] * nodes_[2 * v];
    x[1] += n[k] * nodes_[2 * v + 1];
  }
  return x;
}

std::array<double, 4> MacroMesh::jacobian(index_t c, double xi, double eta) const {
  double n[9], dxi[9], deta[9];
  const int nk = shape(type_, xi, eta, n, dxi, deta);
  std::array<double, 4> j{0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nk; ++k) {
    const index_t v = cell_node(c, k);
    j[0] += dxi[k] * nodes_[2 * v];
    j[1] += deta[k] * nodes_[2 * v];
    j[2] += dxi[k] * nodes_[2 * v + 1];
    j[3] += deta[k] * nodes_[2 * v + 1];
  }
  return j;
}

}  // namespace insem::mesh
