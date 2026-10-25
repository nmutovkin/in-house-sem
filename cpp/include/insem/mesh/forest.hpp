#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "insem/core/config.hpp"
#include "insem/mesh/macro_mesh.hpp"

namespace insem::mesh {

// A computational element: a quadtree cell of macro-cell `tree`. At `level`
// the macro reference square is split into 2^level x 2^level cells and the
// leaf is cell (i, j), i along xi and j along eta.
struct Leaf {
  index_t tree;
  std::int32_t level;
  std::int32_t i;
  std::int32_t j;
};

// The leaf across a face. `leaf` is -1 on the domain boundary. `face` is the
// neighbour's face that touches ours, and `level_diff` = neighbour level -
// our level (negative: the neighbour is coarser and our face hangs on it;
// positive: the neighbour is finer and is one of several across the face).
struct FaceNeighbor {
  index_t leaf = -1;
  int face = -1;
  int level_diff = 0;
};

// Corner connectivity of the leaves: corners[4 * leaf + k] is the vertex at
// local corner k (counter-clockwise from (-1, -1)).
struct LeafVertices {
  index_t n_vertices = 0;
  std::vector<index_t> corners;
};

// A forest of quadtrees over a macro-mesh (docs/DESIGN.md §6). Leaves are
// stored tree by tree, in Morton (Z-order) within each tree, which is the
// space-filling-curve order used for locality and partitioning.
//
// Refinement replaces a leaf by its four children in Morton order, so the
// order is preserved. After `balance()` neighbouring leaves differ by at
// most one level across every face (2:1 balance), which bounds hanging
// faces to one fine level against one coarse face.
class Forest {
 public:
  // Deepest supported level. Lattice coordinates are integers on a
  // 2^kMaxLevel grid per tree, so keys are identical for every level.
  static constexpr int kMaxLevel = 20;

  // Every macro-cell refined `level` times.
  static Forest uniform(std::shared_ptr<const MacroMesh> macro, int level);

  const MacroMesh& macro() const { return *macro_; }
  std::shared_ptr<const MacroMesh> macro_ptr() const { return macro_; }

  index_t n_leaves() const { return static_cast<index_t>(leaves_.size()); }
  const Leaf& leaf(index_t l) const { return leaves_[l]; }
  const std::vector<Leaf>& leaves() const { return leaves_; }

  // Reference box [xi0, xi1] x [eta0, eta1] of a leaf inside its macro-cell.
  std::array<double, 4> leaf_box(index_t l) const;

  // Unique vertices of the leaf corners, shared across leaves and across
  // macro-cell boundaries. Hanging vertices (the midpoint of a coarse face)
  // are vertices of the fine leaves only.
  LeafVertices leaf_vertices() const;

  // The leaf containing a point given in half-lattice coordinates of a tree
  // (integers on a 2^(kMaxLevel + 1) grid; cells are half-open), or -1.
  index_t find_leaf(index_t tree, std::int64_t a, std::int64_t b) const;

  // Neighbour across face f of leaf l (probed at the face midpoint).
  FaceNeighbor face_neighbor(index_t l, int f) const;

  // Position along the shared face, as a coordinate of `to` leaf: maps the
  // point at leaf coordinate t in [-1, 1] along face f of leaf `from` (t runs
  // along xi for faces 0/2 and eta for faces 1/3) to the coordinate along
  // face `to_face` of leaf `to` (same convention). Works across macro-cell
  // edges of either orientation.
  double face_coordinate(index_t from, int f, double t, index_t to, int to_face) const;

  // A new forest where every leaf with mark[l] != 0 is split in four.
  Forest refine(const std::vector<char>& mark) const;

  // Refines until neighbouring leaves differ by at most one level across
  // every face, and by zero levels across faces between trees for which
  // `must_match(tree_a, tree_b)` is true (e.g. fluid-solid interfaces).
  Forest balance(const std::function<bool(index_t, index_t)>& must_match = {}) const;

  int max_level() const;

  // Axisymmetric meshes: the leaf touches the axis with its face 3 (its
  // xi direction is radial and uses Gauss-Lobatto-Jacobi points).
  bool is_axial(index_t l) const { return leaves_[l].i == 0 && macro_->touches_axis(leaves_[l].tree); }

 private:
  Forest(std::shared_ptr<const MacroMesh> macro, std::vector<Leaf> leaves);

  // (tree, level) and (i, j) packed in two words.
  struct Key {
    std::uint64_t tree_level;
    std::uint64_t ij;
    bool operator==(const Key& o) const { return tree_level == o.tree_level && ij == o.ij; }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return static_cast<std::size_t>(k.tree_level * 0x9e3779b97f4a7c15ULL ^ (k.ij + 0x632be59bd9b4e019ULL));
    }
  };
  static Key key(index_t tree, int level, std::int64_t i, std::int64_t j);

  std::shared_ptr<const MacroMesh> macro_;
  std::vector<Leaf> leaves_;
  std::unordered_map<Key, index_t, KeyHash> index_;
};

}  // namespace insem::mesh
