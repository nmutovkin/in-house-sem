#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "insem/core/config.hpp"
#include "insem/mesh/forest.hpp"
#include "insem/ops/dof_map.hpp"

namespace insem::geometry {

// How much metric data an element needs (docs/DESIGN.md §3.1). Element
// blocks are split by kind so that kernels for the common cases read almost
// no geometry data.
enum class GeometryKind : std::uint8_t {
  Rect = 0,    // axis-aligned rectangle: x depends on xi only, z on eta only
  Affine = 1,  // parallelogram: constant Jacobian
  Curved = 2,  // general isoparametric mapping: per-point metric
  Axisym = 3,  // axisymmetric (r, z): per-point metric and r-weighted quadrature (blocks only)
};

std::string to_string(GeometryKind kind);

// Per-point geometry of all leaves, in double precision, with the local
// layout of DofMap ([element][j * ngll + i]).
struct Geometry {
  int ngll = 0;
  index_t n_elements = 0;
  std::vector<double> x;  // first coordinate (x, or r in axisymmetric mode)
  std::vector<double> z;
  // Inverse Jacobian of the leaf mapping: d(xi, eta) / d(x, z).
  std::vector<double> xi_x, xi_z, eta_x, eta_z;
  // Jacobian determinant d(x, z) / d(xi, eta) of the leaf mapping (> 0).
  std::vector<double> jac;
  std::vector<GeometryKind> kind;  // per element
};

// Mapping of one leaf from its reference square [-1, 1]^2 (the macro-cell
// mapping restricted to the leaf box), and its Jacobian
// {dx/dxi, dx/deta, dz/dxi, dz/deta} with respect to the leaf coordinates.
std::array<double, 2> leaf_map(const mesh::Forest& forest, index_t leaf, double xi, double eta);
std::array<double, 4> leaf_jacobian(const mesh::Forest& forest, index_t leaf, double xi, double eta);

// Evaluates the leaf mappings (macro-cell mapping restricted to each leaf box)
// at the tensor-product points ref_xi x ref_eta, each on [-1, 1] with
// ngll entries.
Geometry compute_geometry(const mesh::Forest& forest, const std::vector<double>& ref_xi,
                          const std::vector<double>& ref_eta);

// Same, with axial leaves of an axisymmetric mesh (Forest::is_axial) using
// `axial_xi` (the GLJ points) in the xi direction.
Geometry compute_geometry(const mesh::Forest& forest, const std::vector<double>& ref_xi,
                          const std::vector<double>& ref_eta, const std::vector<double>& axial_xi);

// Coordinates of the unique global points ([0] = x, [1] = z), plus the largest
// mismatch between element-local copies of the same global point, which must
// be at round-off level for a conforming mesh.
struct GlobalCoordinates {
  std::vector<double> x;
  std::vector<double> z;
  double max_mismatch = 0.0;
};
GlobalCoordinates global_coordinates(const ops::DofMap& dofs, const Geometry& geom);

}  // namespace insem::geometry
