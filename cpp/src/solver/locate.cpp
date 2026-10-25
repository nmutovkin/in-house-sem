#include "insem/solver/locate.hpp"

#include <algorithm>
#include <cmath>

namespace insem::solver {

std::optional<PointLocation> locate(const mesh::Forest& forest, const geometry::Geometry& geom,
                                    std::array<double, 2> x, const std::vector<index_t>& candidates) {
  const std::size_t npe = static_cast<std::size_t>(geom.ngll) * geom.ngll;
  for (index_t leaf : candidates) {
    // Cheap rejection with the bounding box of the leaf's GLL points, padded
    // because curved leaves may bulge beyond their points.
    const std::size_t p0 = static_cast<std::size_t>(leaf) * npe;
    const auto [xmin, xmax] = std::minmax_element(geom.x.begin() + p0, geom.x.begin() + p0 + npe);
    const auto [zmin, zmax] = std::minmax_element(geom.z.begin() + p0, geom.z.begin() + p0 + npe);
    const double pad = 0.25 * std::max(*xmax - *xmin, *zmax - *zmin);
    if (x[0] < *xmin - pad || x[0] > *xmax + pad || x[1] < *zmin - pad || x[1] > *zmax + pad) continue;

    double xi = 0.0, eta = 0.0;
    bool converged = false;
    for (int it = 0; it < 30; ++it) {
      const auto y = geometry::leaf_map(forest, leaf, xi, eta);
      const auto j = geometry::leaf_jacobian(forest, leaf, xi, eta);
      const double rx = y[0] - x[0];
      const double rz = y[1] - x[1];
      const double det = j[0] * j[3] - j[1] * j[2];
      const double dxi = (j[3] * rx - j[1] * rz) / det;
      const double deta = (-j[2] * rx + j[0] * rz) / det;
      xi = std::clamp(xi - dxi, -2.0, 2.0);
      eta = std::clamp(eta - deta, -2.0, 2.0);
      if (std::abs(dxi) + std::abs(deta) < 1e-12) {
        converged = true;
        break;
      }
    }
    constexpr double kTol = 1e-9;
    if (converged && std::abs(xi) <= 1.0 + kTol && std::abs(eta) <= 1.0 + kTol) {
      return PointLocation{leaf, std::clamp(xi, -1.0, 1.0), std::clamp(eta, -1.0, 1.0)};
    }
  }
  return std::nullopt;
}

PointWeights point_weights(const mesh::Forest& forest, const basis::LagrangeBasis& basis,
                           const PointLocation& loc) {
  return point_weights(forest, basis, basis, loc);
}

PointWeights point_weights(const mesh::Forest& forest, const basis::LagrangeBasis& basis_xi,
                           const basis::LagrangeBasis& basis_eta, const PointLocation& loc) {
  const int n = basis_xi.size();
  const auto lx = basis_xi.values(loc.xi);
  const auto ly = basis_eta.values(loc.eta);
  const auto dlx = basis_xi.derivatives(loc.xi);
  const auto dly = basis_eta.derivatives(loc.eta);
  const auto j = geometry::leaf_jacobian(forest, loc.leaf, loc.xi, loc.eta);
  const double det = j[0] * j[3] - j[1] * j[2];
  const double xi_x = j[3] / det, xi_z = -j[1] / det, eta_x = -j[2] / det, eta_z = j[0] / det;
  PointWeights w;
  w.value.resize(static_cast<std::size_t>(n) * n);
  w.d_dx.resize(w.value.size());
  w.d_dz.resize(w.value.size());
  for (int jj = 0; jj < n; ++jj) {
    for (int ii = 0; ii < n; ++ii) {
      const std::size_t k = static_cast<std::size_t>(jj) * n + ii;
      const double d_xi = dlx[ii] * ly[jj];
      const double d_eta = lx[ii] * dly[jj];
      w.value[k] = lx[ii] * ly[jj];
      w.d_dx[k] = d_xi * xi_x + d_eta * eta_x;
      w.d_dz[k] = d_xi * xi_z + d_eta * eta_z;
    }
  }
  return w;
}

}  // namespace insem::solver
