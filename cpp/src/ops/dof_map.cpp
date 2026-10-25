#include "insem/ops/dof_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>

#include "insem/basis/lagrange.hpp"
#include "insem/basis/quadrature.hpp"
#include "insem/core/error.hpp"

namespace insem::ops {
namespace {

using Row = std::vector<std::pair<index_t, double>>;

// Marker for slots whose constraint row is not built yet.
constexpr index_t kPending = std::numeric_limits<index_t>::min();

// Local index of point m (0..n) along the axis of face f.
int face_slot(int f, int m, int n, int ngll) {
  switch (f) {
    case 0: return m;
    case 1: return m * ngll + n;
    case 2: return n * ngll + m;
    default: return m * ngll;
  }
}

}  // namespace

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

  std::vector<index_t> element_of_leaf(forest.n_leaves(), -1);
  for (index_t e = 0; e < d.n_elem_; ++e) {
    INSEM_CHECK(leaves[e] >= 0 && leaves[e] < forest.n_leaves(), "leaf index " << leaves[e] << " out of range");
    element_of_leaf[leaves[e]] = e;
  }
  const mesh::LeafVertices lv = forest.leaf_vertices();

  // 1. Hanging faces (fine side) and hanging vertices.
  struct Hang {
    index_t coarse = -1;  // coarse element
    int face = -1;        // its face
  };
  std::vector<std::array<Hang, 4>> hang(d.n_elem_);
  std::vector<char> hanging_vertex(lv.n_vertices, 0);
  for (index_t e = 0; e < d.n_elem_; ++e) {
    const index_t leaf = leaves[e];
    for (int f = 0; f < 4; ++f) {
      const mesh::FaceNeighbor nb = forest.face_neighbor(leaf, f);
      if (nb.leaf < 0) continue;
      const index_t ce = element_of_leaf[nb.leaf];
      if (ce < 0) {
        INSEM_CHECK(nb.level_diff == 0, "leaves of different domains meet at different refinement levels");
        continue;
      }
      if (nb.level_diff >= 0) continue;
      INSEM_CHECK(nb.level_diff == -1, "forest is not 2:1 balanced (level jump " << -nb.level_diff << ")");
      hang[e][f] = {ce, nb.face};
      const index_t* cv = &lv.corners[4 * static_cast<std::size_t>(nb.leaf)];
      const index_t c0 = cv[mesh::kFaceCorners[nb.face][0]], c1 = cv[mesh::kFaceCorners[nb.face][1]];
      for (int c = 0; c < 2; ++c) {
        const index_t v = lv.corners[4 * static_cast<std::size_t>(leaf) + mesh::kFaceCorners[f][c]];
        if (v != c0 && v != c1) hanging_vertex[v] = 1;
      }
    }
  }

  // 2. Number the unconstrained points.
  std::vector<index_t> vertex_id(lv.n_vertices, -1);
  struct EdgeEntry {
    index_t base;
    int uses;
  };
  std::unordered_map<std::uint64_t, EdgeEntry> edges;
  edges.reserve(2 * static_cast<std::size_t>(d.n_elem_) + 16);
  d.l2g_.assign(n_slots, -1);
  index_t next = 0;
  for (index_t e = 0; e < d.n_elem_; ++e) {
    const index_t* v = &lv.corners[4 * static_cast<std::size_t>(leaves[e])];
    index_t edge_base[4];
    bool edge_forward[4];
    for (int f = 0; f < 4; ++f) {
      const index_t a = v[mesh::kFaceCorners[f][0]];
      const index_t b = v[mesh::kFaceCorners[f][1]];
      edge_forward[f] = a < b;
      if (hang[e][f].coarse >= 0) {
        edge_base[f] = -1;  // constrained by the coarse face
        continue;
      }
      const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) |
                                static_cast<std::uint32_t>(std::max(a, b));
      auto [it, inserted] = edges.try_emplace(key, EdgeEntry{next, 0});
      if (inserted) next += n - 1;
      ++it->second.uses;
      INSEM_CHECK(it->second.uses <= 2, "leaf edge shared by more than two elements");
      edge_base[f] = it->second.base;
    }
    index_t* map = &d.l2g_[static_cast<std::size_t>(e) * npe];
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const bool xi_end = (i == 0 || i == n);
        const bool eta_end = (j == 0 || j == n);
        index_t g;
        if (xi_end && eta_end) {
          const int corner = (j == 0) ? (i == 0 ? 0 : 1) : (i == n ? 2 : 3);
          if (hanging_vertex[v[corner]]) {
            g = kPending;
          } else {
            index_t& id = vertex_id[v[corner]];
            if (id < 0) id = next++;
            g = id;
          }
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
          if (edge_base[face] < 0) {
            g = kPending;
          } else {
            const int pos = edge_forward[face] ? p : n - p;
            g = edge_base[face] + (pos - 1);
          }
        } else {
          g = next++;
        }
        map[j * ngll + i] = g;
      }
    }
  }
  d.n_global_ = next;

  // 3. Constraint rows: a pending slot lies on a hanging face; its value is
  // the coarse face's Lagrange interpolant at the slot's position. Coarse
  // face nodes may themselves be constrained (chains through corners), so
  // rows are resolved recursively.
  const basis::LagrangeBasis lagrange(basis::gll(ngll).points);
  const auto& nodes = lagrange.nodes();
  std::map<std::size_t, Row> memo;
  std::function<Row(std::size_t)> resolve = [&](std::size_t slot) -> Row {
    const index_t m = d.l2g_[slot];
    if (m >= 0) return {{m, 1.0}};
    if (m != kPending) {
      const index_t r = -1 - m;
      Row row;
      for (index_t q = d.row_offsets_[r]; q < d.row_offsets_[r + 1]; ++q) row.push_back({d.row_dofs_[q], d.row_weights_[q]});
      return row;
    }
    const auto it = memo.find(slot);
    if (it != memo.end()) return it->second;
    const index_t e = static_cast<index_t>(slot / npe);
    const int k = static_cast<int>(slot % npe);
    const int i = k % ngll, j = k / ngll;
    int f = -1;
    for (int cand = 0; cand < 4 && f < 0; ++cand) {
      if (hang[e][cand].coarse < 0) continue;
      const bool on_face = (cand == 0 && j == 0) || (cand == 1 && i == n) || (cand == 2 && j == n) || (cand == 3 && i == 0);
      if (on_face) f = cand;
    }
    INSEM_CHECK(f >= 0, "constrained point is not on a hanging face");
    const Hang h = hang[e][f];
    const double t = (f == 0 || f == 2) ? nodes[i] : nodes[j];
    const double tc = forest.face_coordinate(leaves[e], f, t, leaves[h.coarse], h.face);
    const auto w = lagrange.values(tc);
    std::map<index_t, double> acc;
    for (int q = 0; q <= n; ++q) {
      if (std::abs(w[q]) < 1e-14) continue;
      const std::size_t coarse_slot = static_cast<std::size_t>(h.coarse) * npe + face_slot(h.face, q, n, ngll);
      for (const auto& [g, wg] : resolve(coarse_slot)) acc[g] += w[q] * wg;
    }
    Row row(acc.begin(), acc.end());
    memo[slot] = row;
    return row;
  };
  for (std::size_t slot = 0; slot < n_slots; ++slot) {
    if (d.l2g_[slot] != kPending) continue;
    Row row = resolve(slot);
    if (row.size() == 1 && std::abs(row[0].second - 1.0) < 1e-12) {
      d.l2g_[slot] = row[0].first;  // coincides with a coarse node
      continue;
    }
    d.l2g_[slot] = -1 - d.n_rows();
    for (const auto& [g, w] : row) {
      d.row_dofs_.push_back(g);
      d.row_weights_.push_back(w);
    }
    d.row_offsets_.push_back(static_cast<index_t>(d.row_dofs_.size()));
  }

  // 4. Transpose (CSR), contributions in ascending slot order.
  d.t_offsets_.assign(static_cast<std::size_t>(d.n_global_) + 1, 0);
  auto for_each_entry = [&](auto&& fn) {
    for (std::size_t s = 0; s < n_slots; ++s) {
      const index_t m = d.l2g_[s];
      if (m >= 0) {
        fn(s, m, 1.0);
      } else {
        const index_t r = -1 - m;
        for (index_t q = d.row_offsets_[r]; q < d.row_offsets_[r + 1]; ++q) fn(s, d.row_dofs_[q], d.row_weights_[q]);
      }
    }
  };
  for_each_entry([&](std::size_t, index_t g, double) { ++d.t_offsets_[static_cast<std::size_t>(g) + 1]; });
  for (index_t g = 0; g < d.n_global_; ++g) d.t_offsets_[g + 1] += d.t_offsets_[g];
  d.t_slots_.resize(d.t_offsets_.back());
  if (d.has_constraints()) d.t_weights_.resize(d.t_offsets_.back());
  std::vector<index_t> fill(d.t_offsets_.begin(), d.t_offsets_.end() - 1);
  for_each_entry([&](std::size_t s, index_t g, double w) {
    const index_t at = fill[g]++;
    d.t_slots_[at] = static_cast<index_t>(s);
    if (!d.t_weights_.empty()) d.t_weights_[at] = w;
  });

  d.primary_.assign(d.n_global_, -1);
  for (std::size_t s = 0; s < n_slots; ++s) {
    const index_t m = d.l2g_[s];
    if (m >= 0 && d.primary_[m] < 0) d.primary_[m] = static_cast<index_t>(s);
  }
  for (index_t g = 0; g < d.n_global_; ++g) {
    INSEM_CHECK(d.primary_[g] >= 0, "global point " << g << " has no unconstrained element point");
  }
  return d;
}

index_t DofMap::global_index(index_t e, int k) const {
  const index_t m = l2g_[static_cast<std::size_t>(e) * npe_ + k];
  INSEM_CHECK(m >= 0, "local point " << k << " of element " << e << " is constrained by a hanging face");
  return m;
}

std::vector<std::pair<index_t, double>> DofMap::slot_entries(index_t e, int k) const {
  const index_t m = l2g_[static_cast<std::size_t>(e) * npe_ + k];
  if (m >= 0) return {{m, 1.0}};
  std::vector<std::pair<index_t, double>> out;
  const index_t r = -1 - m;
  for (index_t q = row_offsets_[r]; q < row_offsets_[r + 1]; ++q) out.push_back({row_dofs_[q], row_weights_[q]});
  return out;
}

std::vector<index_t> DofMap::multiplicity() const {
  std::vector<index_t> m(n_global_);
  for (index_t g = 0; g < n_global_; ++g) m[g] = t_offsets_[g + 1] - t_offsets_[g];
  return m;
}

}  // namespace insem::ops
