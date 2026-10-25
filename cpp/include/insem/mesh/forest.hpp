#pragma once

#include <array>
#include <cstdint>
#include <memory>
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
// M0 provides uniform refinement only; adaptive refinement with 2:1 balance
// and hanging-node constraints is milestone M2.
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
  // macro-cell boundaries. Requires a conforming forest (no hanging corners).
  LeafVertices leaf_vertices() const;

 private:
  Forest(std::shared_ptr<const MacroMesh> macro, std::vector<Leaf> leaves);

  std::shared_ptr<const MacroMesh> macro_;
  std::vector<Leaf> leaves_;
};

}  // namespace insem::mesh
