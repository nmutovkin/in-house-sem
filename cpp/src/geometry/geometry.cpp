#include "insem/geometry/geometry.hpp"

#include <algorithm>
#include <cmath>

#include "insem/core/error.hpp"

namespace insem::geometry {

std::string to_string(GeometryKind kind) {
  switch (kind) {
    case GeometryKind::Rect:
      return "rect";
    case GeometryKind::Affine:
      return "affine";
    case GeometryKind::Curved:
      return "curved";
  }
  return "unknown";
}

Geometry compute_geometry(const mesh::Forest& forest, const std::vector<double>& ref_xi,
                          const std::vector<double>& ref_eta) {
  INSEM_CHECK(ref_xi.size() == ref_eta.size() && ref_xi.size() >= 2,
              "reference point sets must have the same size >= 2");
  const mesh::MacroMesh& macro = forest.macro();
  Geometry g;
  g.ngll = static_cast<int>(ref_xi.size());
  g.n_elements = forest.n_leaves();
  const int ngll = g.ngll;
  const std::size_t npe = static_cast<std::size_t>(ngll) * ngll;
  const std::size_t total = npe * g.n_elements;
  for (auto* v : {&g.x, &g.z, &g.xi_x, &g.xi_z, &g.eta_x, &g.eta_z, &g.jac}) v->resize(total);
  g.kind.resize(g.n_elements);

  // Relative tolerance for detecting constant / diagonal Jacobians.
  constexpr double kTol = 1e-10;

#pragma omp parallel for schedule(static)
  for (index_t e = 0; e < g.n_elements; ++e) {
    const index_t cell = forest.leaf(e).tree;
    const auto box = forest.leaf_box(e);
    const double hx = 0.5 * (box[1] - box[0]);
    const double hy = 0.5 * (box[3] - box[2]);
    double scale = 0.0;
    double deviation = 0.0;
    double off_diagonal = 0.0;
    std::array<double, 4> j_first{};
    for (int jq = 0; jq < ngll; ++jq) {
      for (int iq = 0; iq < ngll; ++iq) {
        const double xi = box[0] + (ref_xi[iq] + 1.0) * hx;
        const double eta = box[2] + (ref_eta[jq] + 1.0) * hy;
        const auto x = macro.map(cell, xi, eta);
        auto jm = macro.jacobian(cell, xi, eta);
        // Chain rule from leaf to macro reference coordinates.
        const std::array<double, 4> j{jm[0] * hx, jm[1] * hy, jm[2] * hx, jm[3] * hy};
        const double det = j[0] * j[3] - j[1] * j[2];
        const std::size_t p = e * npe + static_cast<std::size_t>(jq) * ngll + iq;
        g.x[p] = x[0];
        g.z[p] = x[1];
        g.jac[p] = det;
        g.xi_x[p] = j[3] / det;
        g.xi_z[p] = -j[1] / det;
        g.eta_x[p] = -j[2] / det;
        g.eta_z[p] = j[0] / det;
        if (iq == 0 && jq == 0) j_first = j;
        for (int k = 0; k < 4; ++k) {
          scale = std::max(scale, std::abs(j[k]));
          deviation = std::max(deviation, std::abs(j[k] - j_first[k]));
        }
        off_diagonal = std::max({off_diagonal, std::abs(j[1]), std::abs(j[2])});
      }
    }
    if (deviation > kTol * scale) {
      g.kind[e] = GeometryKind::Curved;
    } else if (off_diagonal > kTol * scale) {
      g.kind[e] = GeometryKind::Affine;
    } else {
      g.kind[e] = GeometryKind::Rect;
    }
  }
  for (double det : g.jac) {
    INSEM_CHECK(det > 0.0, "leaf mapping has a non-positive Jacobian (" << det << ")");
  }
  return g;
}

GlobalCoordinates global_coordinates(const ops::DofMap& dofs, const Geometry& geom) {
  INSEM_CHECK(dofs.n_elements() == geom.n_elements && dofs.ngll() == geom.ngll,
              "dof map and geometry do not describe the same elements");
  GlobalCoordinates out;
  out.x.resize(dofs.n_global());
  out.z.resize(dofs.n_global());
  const auto& offsets = dofs.assembly_offsets();
  const auto& slots = dofs.assembly_slots();
  double mismatch = 0.0;
  for (index_t g = 0; g < dofs.n_global(); ++g) {
    const index_t first = slots[offsets[g]];
    out.x[g] = geom.x[first];
    out.z[g] = geom.z[first];
    for (index_t s = offsets[g] + 1; s < offsets[g + 1]; ++s) {
      mismatch = std::max({mismatch, std::abs(geom.x[slots[s]] - out.x[g]),
                           std::abs(geom.z[slots[s]] - out.z[g])});
    }
  }
  out.max_mismatch = mismatch;
  return out;
}

}  // namespace insem::geometry
