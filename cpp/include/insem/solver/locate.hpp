#pragma once

#include <array>
#include <optional>
#include <vector>

#include "insem/basis/lagrange.hpp"
#include "insem/core/config.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/mesh/forest.hpp"

namespace insem::solver {

// A physical point expressed in the reference coordinates of a leaf.
struct PointLocation {
  index_t leaf = -1;
  double xi = 0.0;
  double eta = 0.0;
};

// Finds the leaf among `candidates` (forest leaf ids) that contains `x`,
// by inverting the leaf mapping with Newton's method. Candidates are tried
// in order, so points on shared faces resolve to the first one. Returns
// nothing if no candidate contains the point.
std::optional<PointLocation> locate(const mesh::Forest& forest, const geometry::Geometry& geom,
                                    std::array<double, 2> x, const std::vector<index_t>& candidates);

// Lagrange interpolation weights of a located point for every local point
// k = j * ngll + i of its leaf: value phi_k(x) and physical gradient.
struct PointWeights {
  std::vector<double> value;
  std::vector<double> d_dx;
  std::vector<double> d_dz;
};
PointWeights point_weights(const mesh::Forest& forest, const basis::LagrangeBasis& basis,
                           const PointLocation& loc);
// Same with different bases along xi and eta (axial leaves use GLJ along xi).
PointWeights point_weights(const mesh::Forest& forest, const basis::LagrangeBasis& basis_xi,
                           const basis::LagrangeBasis& basis_eta, const PointLocation& loc);

}  // namespace insem::solver
