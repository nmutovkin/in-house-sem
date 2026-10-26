#pragma once

#include <vector>

#include "insem/core/config.hpp"
#include "insem/geometry/geometry.hpp"

namespace insem::ops {

// A homogeneous group of elements of one physics domain: same geometry kind
// (and, implicitly, the same physics and order). Each block is processed by
// one kernel instantiation without runtime branching (docs/DESIGN.md §7.3).
//
// metric layout by kind:
//   Rect:   3 per element  (xi_x, eta_z, jac)
//   Affine: 5 per element  (xi_x, xi_z, eta_x, eta_z, jac)
//   Curved: 5 per point    (xi_x, xi_z, eta_x, eta_z, jac), points in local order
// coef0 / coef1: material coefficients per point ([block element][k]):
//   elastic:  lambda, mu
//   acoustic: 1 / rho, (unused)
//
// Axisym blocks (docs/DESIGN.md §2.4) additionally hold, per point:
//   wr:    quadrature weight x Jacobian x r (with the GLJ factor on axial
//          elements, where it stays positive on the axis)
//   hoop:  wr / r, the weight of the hoop term (unused on the axis)
//   r:     radius, exactly 0 on the axis
// and per element `axial` (the element's xi = -1 face is on the axis and
// its xi direction uses GLJ points). metric is 5 per point, as Curved.
// `azimuthal_order` is the Fourier order m in theta of the fields (§2.5):
// m = 0 elastic fields are (u_r, u_z), m = 1 elastic fields (U+, U-, U_z).
struct ElementBlock {
  geometry::GeometryKind kind = geometry::GeometryKind::Curved;
  std::vector<index_t> elements;  // domain-local element indices
  std::vector<real_t> metric;
  std::vector<real_t> coef0;
  std::vector<real_t> coef1;
  std::vector<real_t> wr, hoop, r;
  std::vector<char> axial;
  int azimuthal_order = 0;

  index_t size() const { return static_cast<index_t>(elements.size()); }
};

inline constexpr int metric_stride(geometry::GeometryKind kind) {
  return kind == geometry::GeometryKind::Rect ? 3 : 5;
}

// One Axisym block for all elements of a domain. `wr`, `hoop` and `axial`
// are per domain element ([e][k] and [e]).
ElementBlock build_axisym_block(const std::vector<index_t>& leaves, const geometry::Geometry& geom,
                                const std::vector<double>& coef0, const std::vector<double>& coef1,
                                const std::vector<double>& wr, const std::vector<double>& hoop,
                                const std::vector<char>& axial, int azimuthal_order = 0);

// Splits a (Cartesian) domain into blocks by geometry kind.
//   leaves:    forest leaf of each domain element
//   geom:      geometry of all forest leaves
//   coef0/1:   per-point coefficients of the domain elements ([e][k])
std::vector<ElementBlock> build_blocks(const std::vector<index_t>& leaves,
                                       const geometry::Geometry& geom,
                                       const std::vector<double>& coef0,
                                       const std::vector<double>& coef1);

}  // namespace insem::ops
