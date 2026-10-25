// CPU element kernels: sum-factorised stiffness operators, templated on the
// number of GLL points and the geometry kind (docs/DESIGN.md §3.1, §7.3).

#include "insem/physics/kernels.hpp"

#include <cstddef>

#include "insem/basis/lagrange.hpp"
#include "insem/core/error.hpp"
#include "insem/physics/pointwise.hpp"

namespace insem::physics {
namespace {

using geometry::GeometryKind;

// Metric of block element b at local point k.
template <GeometryKind K, int NPE>
inline void load_metric(const real_t* metric, index_t b, int k, real_t& xi_x, real_t& xi_z,
                        real_t& eta_x, real_t& eta_z, real_t& jac) {
  if constexpr (K == GeometryKind::Rect) {
    const real_t* m = metric + static_cast<std::size_t>(b) * 3;
    xi_x = m[0];
    xi_z = 0;
    eta_x = 0;
    eta_z = m[1];
    jac = m[2];
  } else {
    const real_t* m = metric + (K == GeometryKind::Affine ? static_cast<std::size_t>(b) * 5
                                                          : (static_cast<std::size_t>(b) * NPE + k) * 5);
    xi_x = m[0];
    xi_z = m[1];
    eta_x = m[2];
    eta_z = m[3];
    jac = m[4];
  }
}

// Reference derivatives of a nodal field at point (i, j).
template <int NG>
inline void reference_derivatives(const real_t* D, const real_t* u, int i, int j, real_t& u_xi,
                                  real_t& u_eta) {
  real_t a = 0, b = 0;
  for (int l = 0; l < NG; ++l) {
    a += D[i * NG + l] * u[j * NG + l];
    b += D[j * NG + l] * u[l * NG + i];
  }
  u_xi = a;
  u_eta = b;
}

// Weak divergence at point (i, j): sum_l D[l][i] f_xi(l, j) + D[l][j] f_eta(i, l).
template <int NG>
inline real_t weak_divergence(const real_t* D, const real_t* f_xi, const real_t* f_eta, int i, int j) {
  real_t s = 0;
  for (int l = 0; l < NG; ++l) {
    s += D[l * NG + i] * f_xi[j * NG + l];
    s += D[l * NG + j] * f_eta[l * NG + i];
  }
  return s;
}

template <int NG, GeometryKind K>
void elastic_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* u, real_t* f) {
  constexpr int NPE = NG * NG;
  const real_t* D = kb.D.data();
  const real_t* w2 = kb.w2.data();
  const real_t* metric = blk.metric.data();
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const real_t* ux = u + e * 2 * NPE;
    const real_t* uz = ux + NPE;
    const real_t* lambda = blk.coef0.data() + static_cast<std::size_t>(b) * NPE;
    const real_t* mu = blk.coef1.data() + static_cast<std::size_t>(b) * NPE;
    real_t f1x[NPE], f1z[NPE], f2x[NPE], f2z[NPE];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        real_t ux_xi, ux_eta, uz_xi, uz_eta;
        reference_derivatives<NG>(D, ux, i, j, ux_xi, ux_eta);
        reference_derivatives<NG>(D, uz, i, j, uz_xi, uz_eta);
        real_t xi_x, xi_z, eta_x, eta_z, jac;
        load_metric<K, NPE>(metric, b, k, xi_x, xi_z, eta_x, eta_z, jac);
        real_t ux_x, ux_z, uz_x, uz_z;
        if constexpr (K == GeometryKind::Rect) {
          ux_x = ux_xi * xi_x;
          ux_z = ux_eta * eta_z;
          uz_x = uz_xi * xi_x;
          uz_z = uz_eta * eta_z;
        } else {
          physical_gradient(ux_xi, ux_eta, xi_x, xi_z, eta_x, eta_z, ux_x, ux_z);
          physical_gradient(uz_xi, uz_eta, xi_x, xi_z, eta_x, eta_z, uz_x, uz_z);
        }
        real_t sxx, szz, sxz;
        elastic_stress(lambda[k], mu[k], ux_x, ux_z, uz_x, uz_z, sxx, szz, sxz);
        const real_t wj = w2[k] * jac;
        if constexpr (K == GeometryKind::Rect) {
          f1x[k] = wj * sxx * xi_x;
          f1z[k] = wj * sxz * xi_x;
          f2x[k] = wj * sxz * eta_z;
          f2z[k] = wj * szz * eta_z;
        } else {
          reference_flux(wj, sxx, sxz, xi_x, xi_z, eta_x, eta_z, f1x[k], f2x[k]);
          reference_flux(wj, sxz, szz, xi_x, xi_z, eta_x, eta_z, f1z[k], f2z[k]);
        }
      }
    }
    real_t* fx = f + e * 2 * NPE;
    real_t* fz = fx + NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        fx[j * NG + i] = -weak_divergence<NG>(D, f1x, f2x, i, j);
        fz[j * NG + i] = -weak_divergence<NG>(D, f1z, f2z, i, j);
      }
    }
  }
}

template <int NG, GeometryKind K>
void acoustic_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* p, real_t* f) {
  constexpr int NPE = NG * NG;
  const real_t* D = kb.D.data();
  const real_t* w2 = kb.w2.data();
  const real_t* metric = blk.metric.data();
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const real_t* pe = p + e * NPE;
    const real_t* inv_rho = blk.coef0.data() + static_cast<std::size_t>(b) * NPE;
    real_t f1[NPE], f2[NPE];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        real_t p_xi, p_eta;
        reference_derivatives<NG>(D, pe, i, j, p_xi, p_eta);
        real_t xi_x, xi_z, eta_x, eta_z, jac;
        load_metric<K, NPE>(metric, b, k, xi_x, xi_z, eta_x, eta_z, jac);
        const real_t c = w2[k] * jac * inv_rho[k];
        if constexpr (K == GeometryKind::Rect) {
          f1[k] = c * p_xi * xi_x * xi_x;
          f2[k] = c * p_eta * eta_z * eta_z;
        } else {
          real_t p_x, p_z;
          physical_gradient(p_xi, p_eta, xi_x, xi_z, eta_x, eta_z, p_x, p_z);
          reference_flux(c, p_x, p_z, xi_x, xi_z, eta_x, eta_z, f1[k], f2[k]);
        }
      }
    }
    real_t* fe = f + e * NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) fe[j * NG + i] = -weak_divergence<NG>(D, f1, f2, i, j);
    }
  }
}

using KernelFn = void (*)(const KernelBasis&, const ops::ElementBlock&, const real_t*, real_t*);

template <int NG>
KernelFn select_kind(bool elastic, GeometryKind kind) {
  switch (kind) {
    case GeometryKind::Rect:
      return elastic ? &elastic_kernel<NG, GeometryKind::Rect> : &acoustic_kernel<NG, GeometryKind::Rect>;
    case GeometryKind::Affine:
      return elastic ? &elastic_kernel<NG, GeometryKind::Affine> : &acoustic_kernel<NG, GeometryKind::Affine>;
    case GeometryKind::Curved:
      return elastic ? &elastic_kernel<NG, GeometryKind::Curved> : &acoustic_kernel<NG, GeometryKind::Curved>;
  }
  return nullptr;
}

KernelFn select(bool elastic, int ngll, GeometryKind kind) {
  switch (ngll) {
    case 2: return select_kind<2>(elastic, kind);
    case 3: return select_kind<3>(elastic, kind);
    case 4: return select_kind<4>(elastic, kind);
    case 5: return select_kind<5>(elastic, kind);
    case 6: return select_kind<6>(elastic, kind);
    case 7: return select_kind<7>(elastic, kind);
    case 8: return select_kind<8>(elastic, kind);
    case 9: return select_kind<9>(elastic, kind);
    default:
      throw Error("no kernel for " + std::to_string(ngll) + " GLL points (supported: " +
                  std::to_string(kMinNgll) + ".." + std::to_string(kMaxNgll) + ")");
  }
}

}  // namespace

KernelBasis KernelBasis::from_rule(const std::vector<double>& nodes, const std::vector<double>& weights) {
  INSEM_CHECK(nodes.size() == weights.size(), "nodes and weights differ in size");
  KernelBasis kb;
  kb.ngll = static_cast<int>(nodes.size());
  const auto d = basis::LagrangeBasis(nodes).derivative_matrix();
  kb.D.assign(d.begin(), d.end());
  kb.w2.resize(nodes.size() * nodes.size());
  for (int j = 0; j < kb.ngll; ++j) {
    for (int i = 0; i < kb.ngll; ++i) kb.w2[j * kb.ngll + i] = static_cast<real_t>(weights[i] * weights[j]);
  }
  return kb;
}

void elastic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                             const real_t* u_local, real_t* f_local) {
  select(true, basis.ngll, block.kind)(basis, block, u_local, f_local);
}

void acoustic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                              const real_t* p_local, real_t* f_local) {
  select(false, basis.ngll, block.kind)(basis, block, p_local, f_local);
}

}  // namespace insem::physics
