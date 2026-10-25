#pragma once

#include <cstddef>
#include <vector>

#include "insem/core/config.hpp"
#include "insem/mesh/forest.hpp"

namespace insem::ops {

// Map between element-local GLL points and unique global points
// (docs/DESIGN.md §7.3).
//
// Layouts:
//   global field: [component][global point]                (structure of arrays)
//   local field:  [element][component][j * ngll + i]       (element-contiguous)
// where i runs along xi and j along eta.
//
// `gather` copies global values into every element-local slot. `assemble` is
// its transpose: each global point sums the slots that map to it, in
// ascending slot order. It writes every global point exactly once (no
// atomics, no colouring), so results are deterministic for any thread count.
class DofMap {
 public:
  // Numbers the GLL points of a conforming forest. Global indices are
  // assigned in first-touch order while walking the leaves in forest order,
  // which keeps the numbering local along the space-filling curve.
  static DofMap build(const mesh::Forest& forest, int ngll);

  // Same, restricted to the given leaves (e.g. the leaves of one physics
  // domain). Element e of the map is leaf `leaves[e]`.
  static DofMap build(const mesh::Forest& forest, int ngll, const std::vector<index_t>& leaves);

  int ngll() const { return ngll_; }
  int points_per_element() const { return npe_; }
  index_t n_elements() const { return n_elem_; }
  index_t n_global() const { return n_global_; }

  // local_to_global()[e * npe + k]: global index of local point k of element e.
  const std::vector<index_t>& local_to_global() const { return l2g_; }
  index_t global_index(index_t e, int k) const {
    return l2g_[static_cast<std::size_t>(e) * npe_ + k];
  }

  // Transpose map, CSR: the slots e * npe + k of global point g are
  // assembly_slots()[assembly_offsets()[g] .. assembly_offsets()[g + 1]),
  // in ascending order.
  const std::vector<index_t>& assembly_offsets() const { return t_offsets_; }
  const std::vector<index_t>& assembly_slots() const { return t_slots_; }

  // Number of element-local copies of each global point.
  std::vector<index_t> multiplicity() const;

  template <class T>
  void gather(const T* global, T* local, int ncomp) const;

  template <class T>
  void assemble(const T* local, T* global, int ncomp) const;

 private:
  int ngll_ = 0;
  int npe_ = 0;
  index_t n_elem_ = 0;
  index_t n_global_ = 0;
  std::vector<index_t> l2g_;
  std::vector<index_t> t_offsets_;
  std::vector<index_t> t_slots_;
};

template <class T>
void DofMap::gather(const T* global, T* local, int ncomp) const {
  const std::size_t ng = static_cast<std::size_t>(n_global_);
  const std::size_t npe = static_cast<std::size_t>(npe_);
#pragma omp parallel for schedule(static)
  for (index_t e = 0; e < n_elem_; ++e) {
    const index_t* map = &l2g_[static_cast<std::size_t>(e) * npe];
    for (int c = 0; c < ncomp; ++c) {
      const T* g = global + c * ng;
      T* l = local + (static_cast<std::size_t>(e) * ncomp + c) * npe;
      for (std::size_t k = 0; k < npe; ++k) l[k] = g[map[k]];
    }
  }
}

template <class T>
void DofMap::assemble(const T* local, T* global, int ncomp) const {
  const std::size_t ng = static_cast<std::size_t>(n_global_);
  const std::size_t npe = static_cast<std::size_t>(npe_);
#pragma omp parallel for schedule(static)
  for (index_t g = 0; g < n_global_; ++g) {
    for (int c = 0; c < ncomp; ++c) {
      T sum = T(0);
      for (index_t s = t_offsets_[g]; s < t_offsets_[g + 1]; ++s) {
        const std::size_t slot = static_cast<std::size_t>(t_slots_[s]);
        const std::size_t e = slot / npe;
        const std::size_t k = slot - e * npe;
        sum += local[(e * ncomp + c) * npe + k];
      }
      global[c * ng + g] = sum;
    }
  }
}

}  // namespace insem::ops
