#include "insem/sim/simulation.hpp"

#include <utility>
#include <vector>

#include "insem/core/error.hpp"
#include "insem/io/hdf5.hpp"

namespace insem {
namespace {

constexpr const char* kCaseFormat = "insem-case";
constexpr std::int64_t kCaseVersion = 1;

}  // namespace

Simulation::Simulation(std::shared_ptr<const mesh::MacroMesh> mesh, Discretization disc)
    : mesh_(std::move(mesh)), disc_(disc) {
  INSEM_CHECK(mesh_, "simulation needs a mesh");
  INSEM_CHECK(disc_.order >= 1 && disc_.order <= 16,
              "polynomial order " << disc_.order << " not in [1, 16]");
  INSEM_CHECK(disc_.uniform_level >= 0 && disc_.uniform_level <= mesh::Forest::kMaxLevel,
              "uniform refinement level " << disc_.uniform_level << " out of range");
}

void Simulation::setup() {
  if (is_setup()) return;
  const int ngll = disc_.order + 1;
  gll_ = basis::gll(ngll);
  forest_ = mesh::Forest::uniform(mesh_, disc_.uniform_level);
  dofs_ = ops::DofMap::build(*forest_, ngll);
  geometry_ = geometry::compute_geometry(*forest_, gll_->points, gll_->points);
}

void Simulation::require_setup() const {
  INSEM_CHECK(is_setup(), "Simulation::setup() has not been called");
}

const basis::Rule& Simulation::gll() const {
  require_setup();
  return *gll_;
}

const mesh::Forest& Simulation::forest() const {
  require_setup();
  return *forest_;
}

const ops::DofMap& Simulation::dofs() const {
  require_setup();
  return *dofs_;
}

const geometry::Geometry& Simulation::geometry() const {
  require_setup();
  return *geometry_;
}

std::map<std::string, index_t> Simulation::geometry_kind_counts() const {
  require_setup();
  std::map<std::string, index_t> counts{{"rect", 0}, {"affine", 0}, {"curved", 0}};
  for (auto k : geometry_->kind) ++counts[geometry::to_string(k)];
  return counts;
}

void Simulation::save(const std::string& path) const {
  io::H5File f(path, io::H5File::Mode::Truncate);
  f.set_attribute("/", "format", std::string(kCaseFormat));
  f.set_attribute("/", "version", kCaseVersion);
  f.set_attribute("/", "insem_version", std::string(insem::version()));

  const mesh::MacroMesh& m = *mesh_;
  const auto nn = static_cast<std::size_t>(m.n_nodes());
  const auto nc = static_cast<std::size_t>(m.n_cells());
  f.write("/mesh/nodes", m.nodes().data(), {nn, 2});
  f.write("/mesh/cells", m.cells().data(), {nc, static_cast<std::size_t>(m.nodes_per_cell())});
  f.write("/mesh/material", m.materials().data(), {nc});
  std::vector<std::int32_t> boundary;
  boundary.reserve(3 * m.boundary().size());
  for (const auto& bf : m.boundary()) {
    boundary.push_back(bf.cell);
    boundary.push_back(bf.face);
    boundary.push_back(bf.tag);
  }
  f.write("/mesh/boundary", boundary.data(), {m.boundary().size(), 3});
  f.set_attribute("/mesh", "cell_type", mesh::to_string(m.cell_type()));
  f.set_attribute("/mesh", "coords", mesh::to_string(m.coord_system()));

  f.create_group("/discretization");
  f.set_attribute("/discretization", "order", static_cast<std::int64_t>(disc_.order));
  f.set_attribute("/discretization", "uniform_level", static_cast<std::int64_t>(disc_.uniform_level));
}

Simulation Simulation::load(const std::string& path) {
  io::H5File f(path, io::H5File::Mode::Read);
  INSEM_CHECK(f.has_attribute("/", "format") && f.attribute_string("/", "format") == kCaseFormat,
              "'" << path << "' is not an insem case file");
  const std::int64_t version = f.attribute_int("/", "version");
  INSEM_CHECK(version == kCaseVersion,
              "case file version " << version << " is not supported (expected " << kCaseVersion << ")");

  std::vector<std::size_t> dims;
  std::vector<double> nodes = f.read<double>("/mesh/nodes", &dims);
  INSEM_CHECK(dims.size() == 2 && dims[1] == 2, "/mesh/nodes must have shape (n, 2)");
  std::vector<std::int32_t> cells = f.read<std::int32_t>("/mesh/cells");
  std::vector<std::int32_t> material = f.read<std::int32_t>("/mesh/material");
  std::vector<std::int32_t> raw_boundary = f.read<std::int32_t>("/mesh/boundary");
  INSEM_CHECK(raw_boundary.size() % 3 == 0, "/mesh/boundary must have shape (n, 3)");
  std::vector<mesh::BoundaryFace> boundary;
  for (std::size_t i = 0; i < raw_boundary.size(); i += 3) {
    boundary.push_back({raw_boundary[i], raw_boundary[i + 1], raw_boundary[i + 2]});
  }
  auto m = std::make_shared<const mesh::MacroMesh>(
      mesh::cell_type_from_string(f.attribute_string("/mesh", "cell_type")),
      mesh::coord_system_from_string(f.attribute_string("/mesh", "coords")), std::move(nodes),
      std::move(cells), std::vector<int>(material.begin(), material.end()), std::move(boundary));

  Discretization disc;
  disc.order = static_cast<int>(f.attribute_int("/discretization", "order"));
  disc.uniform_level = static_cast<int>(f.attribute_int("/discretization", "uniform_level"));
  return Simulation(std::move(m), disc);
}

}  // namespace insem
