#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "insem/core/config.hpp"

namespace insem::mesh {

// Macro-element (coarse cell) types. Node ordering follows the usual
// convention: corners counter-clockwise starting at (xi, eta) = (-1, -1),
// then (Quad9 only) the midpoints of faces 0..3, then the centre.
enum class CellType : int { Quad4 = 4, Quad9 = 9 };

// Cartesian: coordinates are (x, z). Axisymmetric: coordinates are (r, z)
// with r >= 0 and the symmetry axis at r = 0.
enum class CoordSystem : int { Cartesian = 0, Axisymmetric = 1 };

std::string to_string(CellType type);
std::string to_string(CoordSystem coords);
CellType cell_type_from_string(const std::string& name);
CoordSystem coord_system_from_string(const std::string& name);

// Local faces of a quadrilateral, as (first corner, second corner) in
// counter-clockwise order: face 0 is eta = -1, 1 is xi = +1, 2 is eta = +1,
// 3 is xi = -1.
inline constexpr int kFaceCorners[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

// A tagged face on the domain boundary (free surface, absorbing, axis, ...).
// Tag values are chosen by the model builder.
struct BoundaryFace {
  index_t cell;
  int face;
  int tag;
};

// The conforming coarse mesh that carries the geometry: every material
// interface lies on cell faces. Computational elements are obtained by
// refining its cells (see Forest).
class MacroMesh {
 public:
  // nodes: n_nodes * 2 coordinates; cells: n_cells * nodes_per_cell node
  // indices; material: one region id per cell.
  MacroMesh(CellType type, CoordSystem coords, std::vector<double> nodes,
            std::vector<index_t> cells, std::vector<int> material,
            std::vector<BoundaryFace> boundary = {});

  CellType cell_type() const { return type_; }
  CoordSystem coord_system() const { return coords_; }
  int nodes_per_cell() const { return static_cast<int>(type_); }

  index_t n_nodes() const { return static_cast<index_t>(nodes_.size() / 2); }
  index_t n_cells() const { return static_cast<index_t>(material_.size()); }
  index_t n_edges() const { return static_cast<index_t>(edge_nodes_.size() / 2); }

  std::array<double, 2> node(index_t i) const { return {nodes_[2 * i], nodes_[2 * i + 1]}; }
  index_t cell_node(index_t c, int k) const { return cells_[c * nodes_per_cell() + k]; }
  int material(index_t c) const { return material_[c]; }

  const std::vector<double>& nodes() const { return nodes_; }
  const std::vector<index_t>& cells() const { return cells_; }
  const std::vector<int>& materials() const { return material_; }
  const std::vector<BoundaryFace>& boundary() const { return boundary_; }

  // Topology. Edges are identified by their two corner nodes; the canonical
  // edge direction runs from the lower to the higher node index.
  index_t cell_edge(index_t c, int face) const { return cell_edges_[4 * c + face]; }
  // True when the local face direction (kFaceCorners order) matches the
  // canonical edge direction.
  bool cell_edge_forward(index_t c, int face) const { return cell_edge_forward_[4 * c + face] != 0; }
  // The (up to) two cells sharing an edge; the second is -1 on the boundary.
  std::array<index_t, 2> edge_cells(index_t e) const { return {edge_cells_[2 * e], edge_cells_[2 * e + 1]}; }
  std::array<index_t, 2> edge_nodes(index_t e) const { return {edge_nodes_[2 * e], edge_nodes_[2 * e + 1]}; }
  bool is_boundary_edge(index_t e) const { return edge_cells_[2 * e + 1] < 0; }

  // Isoparametric mapping of cell c from reference (xi, eta) in [-1, 1]^2.
  std::array<double, 2> map(index_t c, double xi, double eta) const;
  // Jacobian {dx/dxi, dx/deta, dz/dxi, dz/deta} of the mapping.
  std::array<double, 4> jacobian(index_t c, double xi, double eta) const;

 private:
  void build_topology();
  void validate() const;

  CellType type_;
  CoordSystem coords_;
  std::vector<double> nodes_;
  std::vector<index_t> cells_;
  std::vector<int> material_;
  std::vector<BoundaryFace> boundary_;

  std::vector<index_t> cell_edges_;
  std::vector<std::uint8_t> cell_edge_forward_;
  std::vector<index_t> edge_nodes_;
  std::vector<index_t> edge_cells_;
};

}  // namespace insem::mesh
