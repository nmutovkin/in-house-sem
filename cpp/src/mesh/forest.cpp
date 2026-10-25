#include "insem/mesh/forest.hpp"

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
    : macro_(std::move(macro)), leaves_(std::move(leaves)) {}

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
