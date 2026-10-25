#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "insem/basis/quadrature.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/mesh/forest.hpp"
#include "insem/mesh/macro_mesh.hpp"
#include "insem/ops/dof_map.hpp"

namespace insem {

// Parameters that turn the macro-mesh into computational elements.
struct Discretization {
  int order = 4;          // polynomial degree N (N + 1 GLL points per direction)
  int uniform_level = 0;  // uniform quadtree refinement of every macro-cell
};

// The single object model of a run (docs/DESIGN.md §7.1). It holds the
// description of the problem (what goes into the case file); `setup()`
// derives the computational data (forest, numbering, geometry) from it.
//
// M0 skeleton: mesh + discretisation only. Materials, forcings, observers,
// integrator and backend are added in M1.
class Simulation {
 public:
  Simulation(std::shared_ptr<const mesh::MacroMesh> mesh, Discretization disc);

  const mesh::MacroMesh& mesh() const { return *mesh_; }
  std::shared_ptr<const mesh::MacroMesh> mesh_ptr() const { return mesh_; }
  const Discretization& discretization() const { return disc_; }

  // Builds forest, dof map and geometry. Idempotent.
  void setup();
  bool is_setup() const { return forest_.has_value(); }

  const basis::Rule& gll() const;
  const mesh::Forest& forest() const;
  const ops::DofMap& dofs() const;
  const geometry::Geometry& geometry() const;

  // Number of elements per geometry kind ("rect", "affine", "curved").
  std::map<std::string, index_t> geometry_kind_counts() const;

  // Case file: a serialisation of the problem description, independent of
  // backend, rank count and derived data.
  void save(const std::string& path) const;
  static Simulation load(const std::string& path);

 private:
  void require_setup() const;

  std::shared_ptr<const mesh::MacroMesh> mesh_;
  Discretization disc_;

  std::optional<basis::Rule> gll_;
  std::optional<mesh::Forest> forest_;
  std::optional<ops::DofMap> dofs_;
  std::optional<geometry::Geometry> geometry_;
};

}  // namespace insem
