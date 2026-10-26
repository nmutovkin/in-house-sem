#pragma once

#include <vector>

#include "insem/core/config.hpp"
#include "insem/ops/element_block.hpp"

namespace insem::physics {

// Supported number of GLL points per direction (polynomial order 1..8).
inline constexpr int kMinNgll = 2;
inline constexpr int kMaxNgll = 9;

// 1D basis data used by the element kernels.
struct KernelBasis {
  int ngll = 0;
  std::vector<real_t> D;   // D[i * ngll + l] = l_l'(x_i)
  std::vector<real_t> w2;  // w2[j * ngll + i] = w_i w_j
  std::vector<real_t> Dj;  // derivative matrix on the GLJ(0,1) points (axial elements)

  // GLL nodes and weights (double) to kernel data (real_t); Dj from the
  // GLJ points of the same size.
  static KernelBasis from_rule(const std::vector<double>& nodes, const std::vector<double>& weights);
};

// Internal forces f = -K u of one block (CPU backend), with element-local
// layouts [element][component][k]:
//   elastic:  u, f have 2 components (x, z; axisymmetric: r, z), or in
//             azimuthal order 1 three (U+, U-, U_z, docs/DESIGN.md §2.5)
//   acoustic: u = p, f have 1 component
// Only the elements of the block are written.
void elastic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                             const real_t* u_local, real_t* f_local);
void acoustic_internal_forces(const KernelBasis& basis, const ops::ElementBlock& block,
                              const real_t* p_local, real_t* f_local);

}  // namespace insem::physics
