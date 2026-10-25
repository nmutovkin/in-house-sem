#pragma once

// Point-wise physics shared by the CPU and CUDA backends (docs/DESIGN.md §7.3).
// Everything here acts on one quadrature point and must stay free of
// allocation, exceptions and library calls other than basic arithmetic.

#include "insem/core/config.hpp"

namespace insem::physics {

// Isotropic plane-strain stress from the displacement gradient.
template <class T>
INSEM_HD inline void elastic_stress(T lambda, T mu, T ux_x, T ux_z, T uz_x, T uz_z, T& sxx, T& szz,
                                    T& sxz) {
  const T div = ux_x + uz_z;
  sxx = lambda * div + T(2) * mu * ux_x;
  szz = lambda * div + T(2) * mu * uz_z;
  sxz = mu * (ux_z + uz_x);
}

// Physical gradient from reference derivatives: (a_x, a_z) = J^-T (a_xi, a_eta).
template <class T>
INSEM_HD inline void physical_gradient(T a_xi, T a_eta, T xi_x, T xi_z, T eta_x, T eta_z, T& a_x,
                                       T& a_z) {
  a_x = a_xi * xi_x + a_eta * eta_x;
  a_z = a_xi * xi_z + a_eta * eta_z;
}

// Flux of a vector (q_x, q_z) through the reference directions, weighted by
// the quadrature weight times Jacobian: f_xi = wj (q . grad xi),
// f_eta = wj (q . grad eta). Contracting these with D^T gives the weak
// divergence term  -int grad(phi) . q.
template <class T>
INSEM_HD inline void reference_flux(T wj, T q_x, T q_z, T xi_x, T xi_z, T eta_x, T eta_z, T& f_xi,
                                    T& f_eta) {
  f_xi = wj * (q_x * xi_x + q_z * xi_z);
  f_eta = wj * (q_x * eta_x + q_z * eta_z);
}

// Stacey absorbing condition, elastic: traction t = -C v with
// C = w rho [vp n n^T + vs (I - n n^T)], returned as (cxx, czz, cxz).
// `w` is the boundary quadrature weight times the surface Jacobian.
template <class T>
INSEM_HD inline void stacey_elastic(T w, T rho, T vp, T vs, T nx, T nz, T& cxx, T& czz, T& cxz) {
  const T a = w * rho * vp;
  const T b = w * rho * vs;
  cxx = a * nx * nx + b * (T(1) - nx * nx);
  czz = a * nz * nz + b * (T(1) - nz * nz);
  cxz = (a - b) * nx * nz;
}

// Stacey absorbing condition, acoustic pressure formulation: the boundary
// term is -c dp/dt with c = w / (rho vp).
template <class T>
INSEM_HD inline T stacey_acoustic(T w, T rho, T vp) {
  return w / (rho * vp);
}

}  // namespace insem::physics
