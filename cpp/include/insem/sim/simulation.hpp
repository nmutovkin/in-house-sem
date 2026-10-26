#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "insem/basis/quadrature.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/mesh/forest.hpp"
#include "insem/mesh/macro_mesh.hpp"
#include "insem/ops/dof_map.hpp"
#include "insem/sim/description.hpp"

namespace insem {

namespace solver {
class Solver;
}

// Refine leaves that intersect a box to at least `level`.
struct RefinementRegion {
  double xmin, xmax, zmin, zmax;
  int level;
};

// Parameters that turn the macro-mesh into computational elements
// (docs/DESIGN.md §6.2): uniform refinement, then adaptive refinement until
// every leaf satisfies h <= order * lambda_min / points_per_wavelength
// (lambda_min = slowest wave speed / f_max; skipped when f_max = 0) and the
// refinement regions, then 2:1 balance with equal levels across
// fluid-solid faces.
//
// azimuthal_order (axisymmetric meshes only, docs/DESIGN.md §2.5) selects the
// Fourier order m of the fields in theta: 0 for axisymmetric fields
// (monopole), 1 for fields varying as cos(theta) / sin(theta) (dipole).
struct Discretization {
  int order = 4;          // polynomial degree N (N + 1 GLL points per direction)
  int uniform_level = 0;  // uniform quadtree refinement of every macro-cell
  double f_max = 0.0;     // highest frequency to resolve (Hz); 0 = no wavelength criterion
  double points_per_wavelength = 5.0;
  int max_level = 12;     // adaptive refinement stops at this level
  std::vector<RefinementRegion> regions;
  int azimuthal_order = 0;

  bool adaptive() const { return f_max > 0.0 || !regions.empty(); }
};

// The single object model of a run (docs/DESIGN.md §7.1). It holds the
// problem description (everything in the case file): mesh, discretisation,
// materials, boundary conditions, sources, receivers and time settings.
// `setup()` derives the discretisation (forest, numbering, geometry);
// `solver()` builds the time-stepping state from it on first use. Changing
// the description discards an existing solver.
class Simulation {
 public:
  Simulation(std::shared_ptr<const mesh::MacroMesh> mesh, Discretization disc);
  ~Simulation();
  Simulation(Simulation&&) noexcept;
  Simulation& operator=(Simulation&&) noexcept;

  const mesh::MacroMesh& mesh() const { return *mesh_; }
  std::shared_ptr<const mesh::MacroMesh> mesh_ptr() const { return mesh_; }
  const Discretization& discretization() const { return disc_; }

  // ---- problem description
  void set_material(int region, const Material& material);
  void set_boundary(int tag, BoundaryType type);
  void add_source(const Source& source);
  void clear_sources();
  void add_receiver(const Receiver& receiver);
  void clear_receivers();
  void set_time(const TimeConfig& time);

  const std::map<int, Material>& materials() const { return materials_; }
  const std::map<int, BoundaryType>& boundaries() const { return boundaries_; }
  const std::vector<Source>& sources() const { return sources_; }
  const std::vector<Receiver>& receivers() const { return receivers_; }
  const TimeConfig& time() const { return time_; }

  // ---- derived discretisation
  void setup();
  bool is_setup() const { return forest_.has_value(); }

  const basis::Rule& gll() const;
  const mesh::Forest& forest() const;
  const ops::DofMap& dofs() const;
  const geometry::Geometry& geometry() const;

  // Number of elements per geometry kind ("rect", "affine", "curved").
  std::map<std::string, index_t> geometry_kind_counts() const;

  // ---- time stepping
  solver::Solver& solver();
  bool has_solver() const { return solver_ != nullptr; }
  void run();
  void write_results(const std::string& path);

  // ---- case file: a serialisation of the problem description, independent
  // of backend, rank count and derived data.
  void save(const std::string& path) const;
  static Simulation load(const std::string& path);

 private:
  void require_setup() const;
  void invalidate_solver();
  void invalidate_setup();
  mesh::Forest build_forest() const;

  std::shared_ptr<const mesh::MacroMesh> mesh_;
  Discretization disc_;
  std::map<int, Material> materials_;
  std::map<int, BoundaryType> boundaries_;
  std::vector<Source> sources_;
  std::vector<Receiver> receivers_;
  TimeConfig time_;

  std::optional<basis::Rule> gll_;
  std::optional<mesh::Forest> forest_;
  std::optional<ops::DofMap> dofs_;
  std::optional<geometry::Geometry> geometry_;
  std::unique_ptr<solver::Solver> solver_;
};

}  // namespace insem
