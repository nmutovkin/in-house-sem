#include "insem/ops/element_block.hpp"

#include "insem/core/error.hpp"

namespace insem::ops {

ElementBlock build_axisym_block(const std::vector<index_t>& leaves, const geometry::Geometry& geom,
                                const std::vector<double>& coef0, const std::vector<double>& coef1,
                                const std::vector<double>& wr, const std::vector<double>& hoop,
                                const std::vector<char>& axial, int azimuthal_order) {
  const std::size_t npe = static_cast<std::size_t>(geom.ngll) * geom.ngll;
  INSEM_CHECK(wr.size() == leaves.size() * npe && hoop.size() == wr.size() && axial.size() == leaves.size(),
              "axisymmetric weights do not match the domain size");
  ElementBlock b;
  b.kind = geometry::GeometryKind::Axisym;
  b.axial = axial;
  b.azimuthal_order = azimuthal_order;
  for (std::size_t e = 0; e < leaves.size(); ++e) {
    b.elements.push_back(static_cast<index_t>(e));
    const std::size_t p0 = static_cast<std::size_t>(leaves[e]) * npe;
    for (std::size_t k = 0; k < npe; ++k) {
      const std::size_t p = p0 + k;
      b.metric.insert(b.metric.end(), {static_cast<real_t>(geom.xi_x[p]), static_cast<real_t>(geom.xi_z[p]),
                                       static_cast<real_t>(geom.eta_x[p]), static_cast<real_t>(geom.eta_z[p]),
                                       static_cast<real_t>(geom.jac[p])});
      const std::size_t q = e * npe + k;
      b.coef0.push_back(static_cast<real_t>(coef0[q]));
      b.coef1.push_back(static_cast<real_t>(coef1[q]));
      b.wr.push_back(static_cast<real_t>(wr[q]));
      b.hoop.push_back(static_cast<real_t>(hoop[q]));
      // Axis points (exactly r = 0 by construction of the mesh) get r = 0.
      b.r.push_back(static_cast<real_t>(geom.x[p] <= 0.0 ? 0.0 : geom.x[p]));
    }
  }
  return b;
}

std::vector<ElementBlock> build_blocks(const std::vector<index_t>& leaves,
                                       const geometry::Geometry& geom,
                                       const std::vector<double>& coef0,
                                       const std::vector<double>& coef1) {
  using geometry::GeometryKind;
  const std::size_t npe = static_cast<std::size_t>(geom.ngll) * geom.ngll;
  INSEM_CHECK(coef0.size() == leaves.size() * npe && coef1.size() == leaves.size() * npe,
              "material coefficient arrays do not match the domain size");

  std::vector<ElementBlock> blocks;
  for (GeometryKind kind : {GeometryKind::Rect, GeometryKind::Affine, GeometryKind::Curved}) {
    ElementBlock b;
    b.kind = kind;
    for (std::size_t e = 0; e < leaves.size(); ++e) {
      const index_t leaf = leaves[e];
      if (geom.kind[leaf] != kind) continue;
      b.elements.push_back(static_cast<index_t>(e));
      const std::size_t p0 = static_cast<std::size_t>(leaf) * npe;
      switch (kind) {
        case GeometryKind::Rect:
          b.metric.insert(b.metric.end(), {static_cast<real_t>(geom.xi_x[p0]),
                                           static_cast<real_t>(geom.eta_z[p0]),
                                           static_cast<real_t>(geom.jac[p0])});
          break;
        case GeometryKind::Affine:
          b.metric.insert(b.metric.end(),
                          {static_cast<real_t>(geom.xi_x[p0]), static_cast<real_t>(geom.xi_z[p0]),
                           static_cast<real_t>(geom.eta_x[p0]), static_cast<real_t>(geom.eta_z[p0]),
                           static_cast<real_t>(geom.jac[p0])});
          break;
        case GeometryKind::Axisym:  // not a classification of leaves
        case GeometryKind::Curved:
          for (std::size_t k = 0; k < npe; ++k) {
            const std::size_t p = p0 + k;
            b.metric.insert(b.metric.end(),
                            {static_cast<real_t>(geom.xi_x[p]), static_cast<real_t>(geom.xi_z[p]),
                             static_cast<real_t>(geom.eta_x[p]), static_cast<real_t>(geom.eta_z[p]),
                             static_cast<real_t>(geom.jac[p])});
          }
          break;
      }
      for (std::size_t k = 0; k < npe; ++k) {
        b.coef0.push_back(static_cast<real_t>(coef0[e * npe + k]));
        b.coef1.push_back(static_cast<real_t>(coef1[e * npe + k]));
      }
    }
    if (!b.elements.empty()) blocks.push_back(std::move(b));
  }
  return blocks;
}

}  // namespace insem::ops
