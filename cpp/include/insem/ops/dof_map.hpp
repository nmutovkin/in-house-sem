#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "insem/core/config.hpp"
#include "insem/mesh/forest.hpp"

namespace insem::ops {

// Map between element-local GLL points and unique global points
// (docs/DESIGN.md §6.3, §7.3).
//
// Layouts:
//   global field: [component][global point]                (structure of arrays)
//   local field:  [element][component][j * ngll + i]       (element-contiguous)
// where i runs along xi and j along eta.
//
// Most local points ("slots") map directly to one global point. Slots on the
// fine side of a coarse-fine face (hanging faces of a 2:1 balanced forest)
// are constrained: their value is the coarse face's polynomial evaluated
// there, a weighted sum of global points. `gather` applies these
// constraints (u_local = C u) and `assemble` their transpose (f = C^T f_local),
// so element kernels never see hanging nodes.
//
// `assemble` writes every global point exactly once, summing its
// contributions in ascending slot order (no atomics, no colouring), so
// results are deterministic for any thread count.
class DofMap {
 public:
  // Numbers the GLL points of a (2:1 balanced) forest. Global indices are
  // assigned in first-touch order while walking the leaves in forest order,
  // which keeps the numbering local along the space-filling curve.
  static DofMap build(const mesh::Forest& forest, int ngll);

  // Same, restricted to the given leaves (e.g. the leaves of one physics
  // domain). Element e of the map is leaf `leaves[e]`. Faces to leaves
  // outside the set must be conforming.
  static DofMap build(const mesh::Forest& forest, int ngll, const std::vector<index_t>& leaves);

  int ngll() const { return ngll_; }
  int points_per_element() const { return npe_; }
  index_t n_elements() const { return n_elem_; }
  index_t n_global() const { return n_global_; }

  // local_to_global()[e * npe + k]: the global index of local point k of
  // element e, or -1 - r for a slot constrained by row r.
  const std::vector<index_t>& local_to_global() const { return l2g_; }
  bool is_constrained(index_t e, int k) const { return l2g_[static_cast<std::size_t>(e) * npe_ + k] < 0; }
  // Global index of an unconstrained slot (throws for a constrained one).
  index_t global_index(index_t e, int k) const;
  // (global point, weight) pairs that define a slot's value.
  std::vector<std::pair<index_t, double>> slot_entries(index_t e, int k) const;

  bool has_constraints() const { return n_rows() > 0; }
  index_t n_rows() const { return static_cast<index_t>(row_offsets_.size()) - 1; }

  // An unconstrained slot e * npe + k that maps to global point g.
  index_t primary_slot(index_t g) const { return primary_[g]; }

  // Transpose map, CSR: the contributions to global point g are slots
  // assembly_slots()[assembly_offsets()[g] .. assembly_offsets()[g + 1]),
  // in ascending order, with weights assembly_weights() (empty when the map
  // has no constraints: all weights are 1).
  const std::vector<index_t>& assembly_offsets() const { return t_offsets_; }
  const std::vector<index_t>& assembly_slots() const { return t_slots_; }
  const std::vector<double>& assembly_weights() const { return t_weights_; }

  // Number of contributions to each global point.
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
  std::vector<index_t> row_offsets_{0};
  std::vector<index_t> row_dofs_;
  std::vector<double> row_weights_;
  std::vector<index_t> primary_;
  std::vector<index_t> t_offsets_;
  std::vector<index_t> t_slots_;
  std::vector<double> t_weights_;
};

template <class T>
void DofMap::gather(const T* global, T* local, int ncomp) const {
  const std::size_t ng = static_cast<std::size_t>(n_global_);
  const std::size_t npe = static_cast<std::size_t>(npe_);
  const index_t* ro = row_offsets_.data();
  const index_t* rd = row_dofs_.data();
  const double* rw = row_weights_.data();
#pragma omp parallel for schedule(static)
  for (index_t e = 0; e < n_elem_; ++e) {
    const index_t* map = &l2g_[static_cast<std::size_t>(e) * npe];
    for (int c = 0; c < ncomp; ++c) {
      const T* g = global + c * ng;
      T* l = local + (static_cast<std::size_t>(e) * ncomp + c) * npe;
      for (std::size_t k = 0; k < npe; ++k) {
        const index_t m = map[k];
        if (m >= 0) {
          l[k] = g[m];
        } else {
          const index_t r = -1 - m;
          T s = T(0);
          for (index_t q = ro[r]; q < ro[r + 1]; ++q) s += static_cast<T>(rw[q]) * g[rd[q]];
          l[k] = s;
        }
      }
    }
  }
}

template <class T>
void DofMap::assemble(const T* local, T* global, int ncomp) const {
  const std::size_t ng = static_cast<std::size_t>(n_global_);
  const std::size_t npe = static_cast<std::size_t>(npe_);
  const bool weighted = !t_weights_.empty();
#pragma omp parallel for schedule(static)
  for (index_t g = 0; g < n_global_; ++g) {
    for (int c = 0; c < ncomp; ++c) {
      T sum = T(0);
      for (index_t s = t_offsets_[g]; s < t_offsets_[g + 1]; ++s) {
        const std::size_t slot = static_cast<std::size_t>(t_slots_[s]);
        const std::size_t e = slot / npe;
        const std::size_t k = slot - e * npe;
        const T v = local[(e * ncomp + c) * npe + k];
        sum += weighted ? static_cast<T>(t_weights_[s]) * v : v;
      }
      global[c * ng + g] = sum;
    }
  }
}

}  // namespace insem::ops
