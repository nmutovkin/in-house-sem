#include "insem/ops/dof_map.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "insem/core/error.hpp"

namespace insem::ops {

DofMap DofMap::build(const mesh::Forest& forest, int ngll) {
  std::vector<index_t> all(forest.n_leaves());
  for (index_t l = 0; l < forest.n_leaves(); ++l) all[l] = l;
  return build(forest, ngll, all);
}

DofMap DofMap::build(const mesh::Forest& forest, int ngll, const std::vector<index_t>& leaves) {
  INSEM_CHECK(ngll >= 2, "need at least 2 GLL points per direction, got " << ngll);
  DofMap d;
  d.ngll_ = ngll;
  d.npe_ = ngll * ngll;
  d.n_elem_ = static_cast<index_t>(leaves.size());
  const int n = ngll - 1;
  const std::size_t npe = static_cast<std::size_t>(d.npe_);
  const std::size_t n_slots = static_cast<std::size_t>(d.n_elem_) * npe;
  INSEM_CHECK(n_slots < static_cast<std::size_t>(std::numeric_limits<index_t>::max()),
              "too many element-local points (" << n_slots << ") for 32-bit indices");

  const mesh::LeafVertices lv = forest.leaf_vertices();
  std::vector<index_t> vertex_id(lv.n_vertices, -1);
  // Leaf edge (lo vertex, hi vertex) -> first global index of its n - 1
  // interior points, numbered in canonical direction lo -> hi; plus a use
  // count to detect non-conforming or non-manifold edges.
  struct EdgeEntry {
    index_t base;
    int uses;
  };
  std::unordered_map<std::uint64_t, EdgeEntry> edges;
  edges.reserve(2 * static_cast<std::size_t>(d.n_elem_) + 16);

  d.l2g_.assign(n_slots, -1);
  index_t next = 0;
  for (index_t e = 0; e < d.n_elem_; ++e) {
    const index_t leaf = leaves[e];
    INSEM_CHECK(leaf >= 0 && leaf < forest.n_leaves(), "leaf index " << leaf << " out of range");
    const index_t* v = &lv.corners[4 * static_cast<std::size_t>(leaf)];
    index_t edge_base[4];
    bool edge_forward[4];
    for (int f = 0; f < 4; ++f) {
      const index_t a = v[mesh::kFaceCorners[f][0]];
      const index_t b = v[mesh::kFaceCorners[f][1]];
      const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) |
                                static_cast<std::uint32_t>(std::max(a, b));
      auto [it, inserted] = edges.try_emplace(key, EdgeEntry{next, 0});
      if (inserted) next += n - 1;
      ++it->second.uses;
      INSEM_CHECK(it->second.uses <= 2, "leaf edge shared by more than two elements");
      edge_base[f] = it->second.base;
      edge_forward[f] = a < b;
    }
    index_t* map = &d.l2g_[static_cast<std::size_t>(e) * npe];
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const bool xi_end = (i == 0 || i == n);
        const bool eta_end = (j == 0 || j == n);
        index_t g;
        if (xi_end && eta_end) {
          const int corner = (j == 0) ? (i == 0 ? 0 : 1) : (i == n ? 2 : 3);
          index_t& id = vertex_id[v[corner]];
          if (id < 0) id = next++;
          g = id;
        } else if (xi_end || eta_end) {
          int face;
          int p;  // position 1..n-1 along the face in local direction
          if (j == 0) {
            face = 0;
            p = i;
          } else if (i == n) {
            face = 1;
            p = j;
          } else if (j == n) {
            face = 2;
            p = n - i;
          } else {
            face = 3;
            p = n - j;
          }
          const int pos = edge_forward[face] ? p : n - p;
          g = edge_base[face] + (pos - 1);
        } else {
          g = next++;
        }
        map[j * ngll + i] = g;
      }
    }
  }
  d.n_global_ = next;

  // Transpose (CSR), slots in ascending order per global point.
  d.t_offsets_.assign(static_cast<std::size_t>(d.n_global_) + 1, 0);
  for (index_t g : d.l2g_) ++d.t_offsets_[static_cast<std::size_t>(g) + 1];
  for (index_t g = 0; g < d.n_global_; ++g) d.t_offsets_[g + 1] += d.t_offsets_[g];
  d.t_slots_.resize(n_slots);
  std::vector<index_t> fill(d.t_offsets_.begin(), d.t_offsets_.end() - 1);
  for (std::size_t s = 0; s < n_slots; ++s) {
    d.t_slots_[fill[d.l2g_[s]]++] = static_cast<index_t>(s);
  }
  for (index_t g = 0; g < d.n_global_; ++g) {
    INSEM_CHECK(d.t_offsets_[g + 1] > d.t_offsets_[g], "global point " << g << " is unused");
  }
  return d;
}

std::vector<index_t> DofMap::multiplicity() const {
  std::vector<index_t> m(n_global_);
  for (index_t g = 0; g < n_global_; ++g) m[g] = t_offsets_[g + 1] - t_offsets_[g];
  return m;
}

}  // namespace insem::ops
