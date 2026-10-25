#include "insem/mesh/forest.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "insem/core/error.hpp"

namespace insem::mesh {
namespace {

// Identity of a lattice point, independent of which macro-cell it is seen
// from: a macro node, a position along a macro edge (in canonical edge
// direction), or a point inside one macro-cell.
struct LatticeKey {
  int kind;  // 0 = macro node, 1 = macro edge, 2 = cell interior
  std::int64_t id;
  std::int64_t a;
  std::int64_t b;
  bool operator==(const LatticeKey& o) const {
    return kind == o.kind && id == o.id && a == o.a && b == o.b;
  }
};

struct LatticeKeyHash {
  std::size_t operator()(const LatticeKey& k) const {
    std::uint64_t h = static_cast<std::uint64_t>(k.kind);
    for (std::int64_t v : {k.id, k.a, k.b}) {
      h ^= static_cast<std::uint64_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
  }
};

LatticeKey lattice_key(const MacroMesh& m, index_t cell, std::int64_t a, std::int64_t b) {
  constexpr std::int64_t s = std::int64_t{1} << Forest::kMaxLevel;
  const bool left = a == 0, right = a == s, bottom = b == 0, top = b == s;
  if ((left || right) && (bottom || top)) {
    const int corner = bottom ? (left ? 0 : 1) : (right ? 2 : 3);
    return {0, m.cell_node(cell, corner), 0, 0};
  }
  if (bottom || right || top || left) {
    int face;
    std::int64_t t;  // position along the face in local (counter-clockwise) direction
    if (bottom) {
      face = 0;
      t = a;
    } else if (right) {
      face = 1;
      t = b;
    } else if (top) {
      face = 2;
      t = s - a;
    } else {
      face = 3;
      t = s - b;
    }
    const std::int64_t canonical = m.cell_edge_forward(cell, face) ? t : s - t;
    return {1, m.cell_edge(cell, face), canonical, 0};
  }
  return {2, cell, a, b};
}

// Inverse of the bit interleaving used for Morton order: extracts the bits
// at even positions of `m`.
std::int32_t compact_bits(std::uint64_t m) {
  std::uint64_t x = m & 0x5555555555555555ULL;
  x = (x | (x >> 1)) & 0x3333333333333333ULL;
  x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x >> 4)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x >> 8)) & 0x0000ffff0000ffffULL;
  x = (x | (x >> 16)) & 0x00000000ffffffffULL;
  return static_cast<std::int32_t>(x);
}

}  // namespace

Forest::Forest(std::shared_ptr<const MacroMesh> macro, std::vector<Leaf> leaves)
    : macro_(std::move(macro)), leaves_(std::move(leaves)) {
  index_.reserve(leaves_.size());
  for (std::size_t l = 0; l < leaves_.size(); ++l) {
    const Leaf& lf = leaves_[l];
    index_[key(lf.tree, lf.level, lf.i, lf.j)] = static_cast<index_t>(l);
  }
}

Forest::Key Forest::key(index_t tree, int level, std::int64_t i, std::int64_t j) {
  return {(static_cast<std::uint64_t>(tree) << 8) | static_cast<std::uint64_t>(level),
          (static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint64_t>(j)};
}

int Forest::max_level() const {
  int m = 0;
  for (const Leaf& l : leaves_) m = std::max(m, static_cast<int>(l.level));
  return m;
}

index_t Forest::find_leaf(index_t tree, std::int64_t a, std::int64_t b) const {
  constexpr std::int64_t h = std::int64_t{1} << (kMaxLevel + 1);
  if (a < 0 || b < 0 || a >= h || b >= h) return -1;
  for (int level = 0; level <= kMaxLevel; ++level) {
    const int shift = kMaxLevel + 1 - level;
    const auto it = index_.find(key(tree, level, a >> shift, b >> shift));
    if (it != index_.end()) return it->second;
  }
  return -1;
}

FaceNeighbor Forest::face_neighbor(index_t l, int f) const {
  constexpr std::int64_t h = std::int64_t{1} << (kMaxLevel + 1);
  const Leaf& lf = leaves_[l];
  const std::int64_t size = std::int64_t{1} << (kMaxLevel + 1 - lf.level);
  const std::int64_t a0 = lf.i * size, b0 = lf.j * size;
  const std::int64_t am = a0 + size / 2, bm = b0 + size / 2;
  std::int64_t a, b;
  switch (f) {
    case 0: a = am; b = b0 - 1; break;
    case 1: a = a0 + size; b = bm; break;
    case 2: a = am; b = b0 + size; break;
    default: a = a0 - 1; b = bm; break;
  }
  FaceNeighbor out;
  index_t tree = lf.tree;
  int nface = (f + 2) % 4;
  if (a < 0 || b < 0 || a >= h || b >= h) {
    // Cross the macro edge: position along it in canonical direction.
    const MacroMesh& m = *macro_;
    const index_t e = m.cell_edge(lf.tree, f);
    const auto cells = m.edge_cells(e);
    if (cells[1] < 0) return out;
    const index_t other = cells[0] == lf.tree ? cells[1] : cells[0];
    int of = 0;
    while (m.cell_edge(other, of) != e) ++of;
    const std::int64_t s_local = (f == 0) ? a : (f == 1) ? b : (f == 2) ? h - a : h - b;
    const std::int64_t c = m.cell_edge_forward(lf.tree, f) ? s_local : h - s_local;
    const std::int64_t s2 = m.cell_edge_forward(other, of) ? c : h - c;
    switch (of) {
      case 0: a = s2; b = 0; break;
      case 1: a = h - 1; b = s2; break;
      case 2: a = h - s2; b = h - 1; break;
      default: a = 0; b = h - s2; break;
    }
    tree = other;
    nface = of;
  }
  out.leaf = find_leaf(tree, a, b);
  INSEM_CHECK(out.leaf >= 0, "forest has no leaf at a face-neighbour position");
  out.face = nface;
  out.level_diff = leaves_[out.leaf].level - lf.level;
  return out;
}

double Forest::face_coordinate(index_t from, int f, double t, index_t to, int to_face) const {
  // Tree-local coordinate u in [0, 1] along the face axis.
  auto axis_range = [this](index_t l, int face) {
    const Leaf& lf = leaves_[l];
    const double n = static_cast<double>(std::int64_t{1} << lf.level);
    const std::int64_t idx = (face == 0 || face == 2) ? lf.i : lf.j;
    return std::array<double, 2>{idx / n, (idx + 1) / n};
  };
  const auto r = axis_range(from, f);
  const double u = r[0] + 0.5 * (t + 1.0) * (r[1] - r[0]);
  double u2 = u;
  const Leaf& a = leaves_[from];
  const Leaf& b = leaves_[to];
  if (a.tree != b.tree) {
    const MacroMesh& m = *macro_;
    const double s = (f == 0 || f == 1) ? u : 1.0 - u;  // counter-clockwise parameter
    const double c = m.cell_edge_forward(a.tree, f) ? s : 1.0 - s;
    const double s2 = m.cell_edge_forward(b.tree, to_face) ? c : 1.0 - c;
    u2 = (to_face == 0 || to_face == 1) ? s2 : 1.0 - s2;
  }
  const auto r2 = axis_range(to, to_face);
  return -1.0 + 2.0 * (u2 - r2[0]) / (r2[1] - r2[0]);
}

Forest Forest::refine(const std::vector<char>& mark) const {
  INSEM_CHECK(mark.size() == leaves_.size(), "refinement marks do not match the leaves");
  std::vector<Leaf> out;
  out.reserve(leaves_.size() + 3 * static_cast<std::size_t>(std::count(mark.begin(), mark.end(), 1)));
  for (std::size_t l = 0; l < leaves_.size(); ++l) {
    const Leaf& lf = leaves_[l];
    if (!mark[l]) {
      out.push_back(lf);
      continue;
    }
    INSEM_CHECK(lf.level < kMaxLevel, "cannot refine beyond level " << kMaxLevel);
    for (int c = 0; c < 4; ++c) {  // Morton order of the children
      out.push_back({lf.tree, lf.level + 1, 2 * lf.i + (c & 1), 2 * lf.j + (c >> 1)});
    }
  }
  return Forest(macro_, std::move(out));
}

Forest Forest::balance(const std::function<bool(index_t, index_t)>& must_match) const {
  Forest current = *this;
  for (;;) {
    std::vector<char> mark(current.leaves_.size(), 0);
    bool any = false;
    for (index_t l = 0; l < current.n_leaves(); ++l) {
      const Leaf& lf = current.leaves_[l];
      for (int f = 0; f < 4; ++f) {
        const FaceNeighbor nb = current.face_neighbor(l, f);
        if (nb.leaf < 0 || nb.level_diff >= 0) continue;
        const index_t other_tree = current.leaves_[nb.leaf].tree;
        const bool strict = must_match && other_tree != lf.tree && must_match(lf.tree, other_tree);
        if (nb.level_diff < -1 || strict) {
          mark[nb.leaf] = 1;
          any = true;
        }
      }
    }
    if (!any) return current;
    current = current.refine(mark);
  }
}

Forest Forest::uniform(std::shared_ptr<const MacroMesh> macro, int level) {
  INSEM_CHECK(macro, "forest needs a macro-mesh");
  INSEM_CHECK(level >= 0 && level <= kMaxLevel,
              "refinement level " << level << " not in [0, " << kMaxLevel << "]");
  const std::uint64_t per_tree = std::uint64_t{1} << (2 * level);
  const std::uint64_t total = per_tree * static_cast<std::uint64_t>(macro->n_cells());
  INSEM_CHECK(total < (std::uint64_t{1} << 31), "uniform refinement to level " << level
                                                    << " gives too many leaves (" << total << ")");
  std::vector<Leaf> leaves;
  leaves.reserve(static_cast<std::size_t>(total));
  for (index_t t = 0; t < macro->n_cells(); ++t) {
    for (std::uint64_t m = 0; m < per_tree; ++m) {
      leaves.push_back({t, level, compact_bits(m), compact_bits(m >> 1)});
    }
  }
  return Forest(std::move(macro), std::move(leaves));
}

std::array<double, 4> Forest::leaf_box(index_t l) const {
  const Leaf& lf = leaves_[l];
  const double h = 2.0 / static_cast<double>(std::int64_t{1} << lf.level);
  return {-1.0 + lf.i * h, -1.0 + (lf.i + 1) * h, -1.0 + lf.j * h, -1.0 + (lf.j + 1) * h};
}

LeafVertices Forest::leaf_vertices() const {
  LeafVertices out;
  out.corners.resize(4 * leaves_.size());
  std::unordered_map<LatticeKey, index_t, LatticeKeyHash> ids;
  ids.reserve(leaves_.size() + macro_->n_nodes());
  constexpr int kCornerDi[4] = {0, 1, 1, 0};
  constexpr int kCornerDj[4] = {0, 0, 1, 1};
  for (std::size_t l = 0; l < leaves_.size(); ++l) {
    const Leaf& lf = leaves_[l];
    const std::int64_t step = std::int64_t{1} << (kMaxLevel - lf.level);
    for (int k = 0; k < 4; ++k) {
      const LatticeKey key = lattice_key(*macro_, lf.tree, (lf.i + kCornerDi[k]) * step,
                                         (lf.j + kCornerDj[k]) * step);
      auto [it, inserted] = ids.try_emplace(key, out.n_vertices);
      if (inserted) ++out.n_vertices;
      out.corners[4 * l + k] = it->second;
    }
  }
  return out;
}

}  // namespace insem::mesh
