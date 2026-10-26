// CPU element kernels: sum-factorised stiffness operators, templated on the
// number of GLL points and the geometry kind (docs/DESIGN.md §3.1, §7.3).

#include "insem/physics/kernels.hpp"

#include <cstddef>

#include "insem/basis/lagrange.hpp"
#include "insem/basis/quadrature.hpp"
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

// ---- axisymmetric (r, z) kernels (docs/DESIGN.md §2.4)
//
// Weak form per radian: integrals carry r (folded into wr). Elastic adds the
// hoop strain u_r / r and stress sigma_tt; on the axis u_r / r -> du_r/dr,
// and the hoop term's test side uses the same limit, so the operator stays
// symmetric: at an axis point q the term wr_q sigma_tt dphi_a/dr (q) is
// spread over the nodes of its row and column.

// Reference derivatives with separate matrices along xi (Dx) and eta (Dy).
template <int NG>
inline void reference_derivatives_xy(const real_t* Dx, const real_t* Dy, const real_t* u, int i, int j,
                                     real_t& u_xi, real_t& u_eta) {
  real_t a = 0, b = 0;
  for (int l = 0; l < NG; ++l) {
    a += Dx[i * NG + l] * u[j * NG + l];
    b += Dy[j * NG + l] * u[l * NG + i];
  }
  u_xi = a;
  u_eta = b;
}

template <int NG>
inline real_t weak_divergence_xy(const real_t* Dx, const real_t* Dy, const real_t* f_xi, const real_t* f_eta,
                                 int i, int j) {
  real_t s = 0;
  for (int l = 0; l < NG; ++l) {
    s += Dx[l * NG + i] * f_xi[j * NG + l];
    s += Dy[l * NG + j] * f_eta[l * NG + i];
  }
  return s;
}

// Spreads the axis-point terms g_j (weight x stress-like factor at axis
// point (0, j)) of an axial element over the nodes through dphi_a/dr:
// f_a -= sum_j g_j dphi_a/dr (0, j), the test-side limit of phi_a / r.
template <int NG>
inline void spread_axis_term(const real_t* Dx, const real_t* Dy, const real_t* m, const real_t* g, real_t* f) {
  // dphi_a/dr at axis point (0, jq) = xi_r Dx[0][ia] (row jq) + eta_r Dy[jq][ja] (column 0).
  for (int jq = 0; jq < NG; ++jq) {
    const int kq = jq * NG;
    const real_t xi_r = m[5 * kq], eta_r = m[5 * kq + 2];
    for (int ia = 0; ia < NG; ++ia) f[jq * NG + ia] -= g[jq] * xi_r * Dx[ia];
    for (int ja = 0; ja < NG; ++ja) f[ja * NG] -= g[jq] * eta_r * Dy[jq * NG + ja];
  }
}

template <int NG>
void elastic_axisym_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* u, real_t* f) {
  constexpr int NPE = NG * NG;
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const bool axial = blk.axial[b] != 0;
    const real_t* Dx = axial ? kb.Dj.data() : kb.D.data();
    const real_t* Dy = kb.D.data();
    const real_t* ur = u + e * 2 * NPE;
    const real_t* uz = ur + NPE;
    const std::size_t p0 = static_cast<std::size_t>(b) * NPE;
    const real_t* m = blk.metric.data() + p0 * 5;
    real_t f1r[NPE], f2r[NPE], f1z[NPE], f2z[NPE], hoop_direct[NPE], axis_g[NG];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        const std::size_t p = p0 + k;
        real_t ur_xi, ur_eta, uz_xi, uz_eta;
        reference_derivatives_xy<NG>(Dx, Dy, ur, i, j, ur_xi, ur_eta);
        reference_derivatives_xy<NG>(Dx, Dy, uz, i, j, uz_xi, uz_eta);
        const real_t xi_r = m[5 * k], xi_z = m[5 * k + 1], eta_r = m[5 * k + 2], eta_z = m[5 * k + 3];
        real_t ur_r, ur_z, uz_r, uz_z;
        physical_gradient(ur_xi, ur_eta, xi_r, xi_z, eta_r, eta_z, ur_r, ur_z);
        physical_gradient(uz_xi, uz_eta, xi_r, xi_z, eta_r, eta_z, uz_r, uz_z);
        const real_t r = blk.r[p];
        const bool on_axis = r == real_t(0);
        const real_t e_tt = on_axis ? ur_r : ur[k] / r;
        const real_t lambda = blk.coef0[p], mu = blk.coef1[p];
        const real_t tr = ur_r + uz_z + e_tt;
        const real_t srr = lambda * tr + 2 * mu * ur_r;
        const real_t szz = lambda * tr + 2 * mu * uz_z;
        const real_t stt = lambda * tr + 2 * mu * e_tt;
        const real_t srz = mu * (ur_z + uz_r);
        const real_t w = blk.wr[p];
        reference_flux(w, srr, srz, xi_r, xi_z, eta_r, eta_z, f1r[k], f2r[k]);
        reference_flux(w, srz, szz, xi_r, xi_z, eta_r, eta_z, f1z[k], f2z[k]);
        if (axial && i == 0) {
          hoop_direct[k] = 0;
          axis_g[j] = w * stt;
        } else {
          hoop_direct[k] = blk.hoop[p] * stt;
        }
      }
    }
    real_t* fr = f + e * 2 * NPE;
    real_t* fz = fr + NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        fr[k] = -(weak_divergence_xy<NG>(Dx, Dy, f1r, f2r, i, j) + hoop_direct[k]);
        fz[k] = -weak_divergence_xy<NG>(Dx, Dy, f1z, f2z, i, j);
      }
    }
    if (axial) spread_axis_term<NG>(Dx, Dy, m, axis_g, fr);
  }
}

template <int NG>
void acoustic_axisym_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* p_in, real_t* f) {
  constexpr int NPE = NG * NG;
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const real_t* Dx = blk.axial[b] ? kb.Dj.data() : kb.D.data();
    const real_t* Dy = kb.D.data();
    const real_t* pe = p_in + e * NPE;
    const std::size_t p0 = static_cast<std::size_t>(b) * NPE;
    const real_t* m = blk.metric.data() + p0 * 5;
    real_t f1[NPE], f2[NPE];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        real_t p_xi, p_eta;
        reference_derivatives_xy<NG>(Dx, Dy, pe, i, j, p_xi, p_eta);
        const real_t xi_r = m[5 * k], xi_z = m[5 * k + 1], eta_r = m[5 * k + 2], eta_z = m[5 * k + 3];
        real_t p_r, p_z;
        physical_gradient(p_xi, p_eta, xi_r, xi_z, eta_r, eta_z, p_r, p_z);
        reference_flux(blk.wr[p0 + k] * blk.coef0[p0 + k], p_r, p_z, xi_r, xi_z, eta_r, eta_z, f1[k], f2[k]);
      }
    }
    real_t* fe = f + e * NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) fe[j * NG + i] = -weak_divergence_xy<NG>(Dx, Dy, f1, f2, i, j);
    }
  }
}

// ---- azimuthal order 1 (docs/DESIGN.md §2.5)
//
// Fields u_r = U_r cos, u_theta = U_theta sin, u_z = U_z cos (theta), stored
// as U+ = (U_r + U_theta) / sqrt 2, U- = (U_r - U_theta) / sqrt 2, U_z. On
// the axis U+ = U_z = 0, and the 1/r terms take their limits there:
//   e_rr = d_r U_r, e_zz = d_z U_z, e_tt = sqrt2 U+ / r, g_rz = d_z U_r + d_r U_z,
//   g_rt = d_r U_theta - sqrt2 U+ / r, g_tz = d_z U_theta - U_z / r,
// with U+ / r -> d_r U+ and U_z / r -> d_r U_z on the axis, on the trial and
// test sides alike. The weak form is per pi (the theta integral of cos^2).
template <int NG>
void elastic_axisym1_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* u, real_t* f) {
  constexpr int NPE = NG * NG;
  const real_t s2 = real_t(1.41421356237309504880);
  const real_t is2 = real_t(0.70710678118654752440);
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const bool axial = blk.axial[b] != 0;
    const real_t* Dx = axial ? kb.Dj.data() : kb.D.data();
    const real_t* Dy = kb.D.data();
    const real_t* up = u + e * 3 * NPE;
    const real_t* um = up + NPE;
    const real_t* uz = um + NPE;
    const std::size_t p0 = static_cast<std::size_t>(b) * NPE;
    const real_t* m = blk.metric.data() + p0 * 5;
    real_t f1p[NPE], f2p[NPE], f1m[NPE], f2m[NPE], f1z[NPE], f2z[NPE];
    real_t direct_p[NPE], direct_z[NPE], axis_p[NG], axis_z[NG];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        const std::size_t p = p0 + k;
        real_t up_xi, up_eta, um_xi, um_eta, uz_xi, uz_eta;
        reference_derivatives_xy<NG>(Dx, Dy, up, i, j, up_xi, up_eta);
        reference_derivatives_xy<NG>(Dx, Dy, um, i, j, um_xi, um_eta);
        reference_derivatives_xy<NG>(Dx, Dy, uz, i, j, uz_xi, uz_eta);
        const real_t xi_r = m[5 * k], xi_z = m[5 * k + 1], eta_r = m[5 * k + 2], eta_z = m[5 * k + 3];
        real_t up_r, up_z, um_r, um_z, uz_r, uz_z;
        physical_gradient(up_xi, up_eta, xi_r, xi_z, eta_r, eta_z, up_r, up_z);
        physical_gradient(um_xi, um_eta, xi_r, xi_z, eta_r, eta_z, um_r, um_z);
        physical_gradient(uz_xi, uz_eta, xi_r, xi_z, eta_r, eta_z, uz_r, uz_z);
        const real_t r = blk.r[p];
        const bool on_axis = r == real_t(0);
        const real_t up_over_r = on_axis ? up_r : up[k] / r;
        const real_t uz_over_r = on_axis ? uz_r : uz[k] / r;
        const real_t ur_r = (up_r + um_r) * is2, ur_z = (up_z + um_z) * is2;
        const real_t ut_r = (up_r - um_r) * is2, ut_z = (up_z - um_z) * is2;
        const real_t e_tt = s2 * up_over_r;
        const real_t lambda = blk.coef0[p], mu = blk.coef1[p];
        const real_t tr = ur_r + uz_z + e_tt;
        const real_t srr = lambda * tr + 2 * mu * ur_r;
        const real_t szz = lambda * tr + 2 * mu * uz_z;
        const real_t stt = lambda * tr + 2 * mu * e_tt;
        const real_t srz = mu * (ur_z + uz_r);
        const real_t srt = mu * (ut_r - s2 * up_over_r);
        const real_t stz = mu * (ut_z - uz_over_r);
        const real_t w = blk.wr[p];
        // Gradient terms: W_r pairs with (s_rr, s_rz), W_theta with (s_rt, s_tz).
        reference_flux(w, (srr + srt) * is2, (srz + stz) * is2, xi_r, xi_z, eta_r, eta_z, f1p[k], f2p[k]);
        reference_flux(w, (srr - srt) * is2, (srz - stz) * is2, xi_r, xi_z, eta_r, eta_z, f1m[k], f2m[k]);
        reference_flux(w, srz, szz, xi_r, xi_z, eta_r, eta_z, f1z[k], f2z[k]);
        // 1/r terms of the test function: W+ / r and W_z / r.
        const real_t cp = s2 * (stt - srt), cz = -stz;
        if (axial && i == 0) {
          direct_p[k] = direct_z[k] = 0;
          axis_p[j] = w * cp;
          axis_z[j] = w * cz;
        } else {
          direct_p[k] = blk.hoop[p] * cp;
          direct_z[k] = blk.hoop[p] * cz;
        }
      }
    }
    real_t* fp = f + e * 3 * NPE;
    real_t* fm = fp + NPE;
    real_t* fz = fm + NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        fp[k] = -(weak_divergence_xy<NG>(Dx, Dy, f1p, f2p, i, j) + direct_p[k]);
        fm[k] = -weak_divergence_xy<NG>(Dx, Dy, f1m, f2m, i, j);
        fz[k] = -(weak_divergence_xy<NG>(Dx, Dy, f1z, f2z, i, j) + direct_z[k]);
      }
    }
    if (axial) {
      spread_axis_term<NG>(Dx, Dy, m, axis_p, fp);
      spread_axis_term<NG>(Dx, Dy, m, axis_z, fz);
    }
  }
}

// Acoustic, p = P cos(theta): -int (1/rho) [grad W . grad P + W P / r^2] r,
// P = 0 on the axis and P / r -> d_r P there.
template <int NG>
void acoustic_axisym1_kernel(const KernelBasis& kb, const ops::ElementBlock& blk, const real_t* p_in, real_t* f) {
  constexpr int NPE = NG * NG;
  const index_t nb = blk.size();
#pragma omp parallel for schedule(static)
  for (index_t b = 0; b < nb; ++b) {
    const std::size_t e = static_cast<std::size_t>(blk.elements[b]);
    const bool axial = blk.axial[b] != 0;
    const real_t* Dx = axial ? kb.Dj.data() : kb.D.data();
    const real_t* Dy = kb.D.data();
    const real_t* pe = p_in + e * NPE;
    const std::size_t p0 = static_cast<std::size_t>(b) * NPE;
    const real_t* m = blk.metric.data() + p0 * 5;
    real_t f1[NPE], f2[NPE], direct[NPE], axis_g[NG];
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        const std::size_t p = p0 + k;
        real_t p_xi, p_eta;
        reference_derivatives_xy<NG>(Dx, Dy, pe, i, j, p_xi, p_eta);
        const real_t xi_r = m[5 * k], xi_z = m[5 * k + 1], eta_r = m[5 * k + 2], eta_z = m[5 * k + 3];
        real_t p_r, p_z;
        physical_gradient(p_xi, p_eta, xi_r, xi_z, eta_r, eta_z, p_r, p_z);
        const real_t inv_rho = blk.coef0[p];
        reference_flux(blk.wr[p] * inv_rho, p_r, p_z, xi_r, xi_z, eta_r, eta_z, f1[k], f2[k]);
        if (axial && i == 0) {
          direct[k] = 0;
          axis_g[j] = blk.wr[p] * inv_rho * p_r;
        } else {
          direct[k] = blk.hoop[p] * inv_rho * pe[k] / blk.r[p];
        }
      }
    }
    real_t* fe = f + e * NPE;
    for (int j = 0; j < NG; ++j) {
      for (int i = 0; i < NG; ++i) {
        const int k = j * NG + i;
        fe[k] = -(weak_divergence_xy<NG>(Dx, Dy, f1, f2, i, j) + direct[k]);
      }
    }
    if (axial) spread_axis_term<NG>(Dx, Dy, m, axis_g, fe);
  }
}

using KernelFn = void (*)(const KernelBasis&, const ops::ElementBlock&, const real_t*, real_t*);

template <int NG>
KernelFn select_kind(bool elastic, GeometryKind kind, int azimuthal_order) {
  switch (kind) {
    case GeometryKind::Rect:
      return elastic ? &elastic_kernel<NG, GeometryKind::Rect> : &acoustic_kernel<NG, GeometryKind::Rect>;
    case GeometryKind::Affine:
      return elastic ? &elastic_kernel<NG, GeometryKind::Affine> : &acoustic_kernel<NG, GeometryKind::Affine>;
    case GeometryKind::Curved:
      return elastic ? &elastic_kernel<NG, GeometryKind::Curved> : &acoustic_kernel<NG, GeometryKind::Curved>;
    case GeometryKind::Axisym:
      if (azimuthal_order == 1) return elastic ? &elastic_axisym1_kernel<NG> : &acoustic_axisym1_kernel<NG>;
      return elastic ? &elastic_axisym_kernel<NG> : &acoustic_axisym_kernel<NG>;
  }
  return nullptr;
}

KernelFn select(bool elastic, int ngll, GeometryKind kind, int azimuthal_order) {
  INSEM_CHECK(azimuthal_order == 0 || (azimuthal_order == 1 && kind == GeometryKind::Axisym),
              "no kernel for azimuthal order " << azimuthal_order << " on " << geometry::to_string(kind)
                                               << " elements");
  switch (ngll) {
    case 2: return select_kind<2>(elastic, kind, azimuthal_order);
    case 3: return select_kind<3>(elastic, kind, azimuthal_order);
    case 4: return select_kind<4>(elastic, kind, azimuthal_order);
    case 5: return select_kind<5>(elastic, kind, azimuthal_order);
    case 6: return select_kind<6>(elastic, kind, azimuthal_order);
    case 7: return select_kind<7>(elastic, kind, azimuthal_order);
    case 8: return select_kind<8>(elastic, kind, azimuthal_order);
    case 9: return select_kind<9>(elastic, kind, azimuthal_order);
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
  const auto dj = basis::LagrangeBasis(basis::glj01(kb.ngll).points).derivative_matrix();
  kb.Dj.assign(dj.begin(), dj.end());
  kb.w2.resize(nodes.size() * nodes.size());
  for (int j = 0; j < kb.ngll; ++j) {
    for (int i = 0; i < kb.ngll; ++i) kb.w2[j * kb.ngll + i] = static_cast<real_t>(weights[i] * weights[j]);
  }
  return kb;
}

void elastic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                             const real_t* u_local, real_t* f_local) {
  select(true, basis.ngll, block.kind, block.azimuthal_order)(basis, block, u_local, f_local);
}

void acoustic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                              const real_t* p_local, real_t* f_local) {
  select(false, basis.ngll, block.kind, block.azimuthal_order)(basis, block, p_local, f_local);
}

}  // namespace insem::physics
