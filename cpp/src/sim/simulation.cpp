#include "insem/sim/simulation.hpp"

#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "insem/core/error.hpp"
#include "insem/io/hdf5.hpp"
#include "insem/solver/solver.hpp"

namespace insem {
namespace {

constexpr const char* kCaseFormat = "insem-case";
constexpr std::int64_t kCaseVersion = 1;

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + items[i];
  return out;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

}  // namespace

Simulation::Simulation(std::shared_ptr<const mesh::MacroMesh> mesh, Discretization disc)
    : mesh_(std::move(mesh)), disc_(disc) {
  INSEM_CHECK(mesh_, "simulation needs a mesh");
  INSEM_CHECK(disc_.order >= 1 && disc_.order <= 16,
              "polynomial order " << disc_.order << " not in [1, 16]");
  INSEM_CHECK(disc_.uniform_level >= 0 && disc_.uniform_level <= mesh::Forest::kMaxLevel,
              "uniform refinement level " << disc_.uniform_level << " out of range");
}

Simulation::~Simulation() = default;
Simulation::Simulation(Simulation&&) noexcept = default;
Simulation& Simulation::operator=(Simulation&&) noexcept = default;

void Simulation::invalidate_solver() { solver_.reset(); }

void Simulation::set_material(int region, const Material& material) {
  material.validate();
  materials_[region] = material;
  invalidate_solver();
}

void Simulation::set_boundary(int tag, BoundaryType type) {
  boundaries_[tag] = type;
  invalidate_solver();
}

void Simulation::add_source(const Source& source) {
  sources_.push_back(source);
  invalidate_solver();
}

void Simulation::clear_sources() {
  sources_.clear();
  invalidate_solver();
}

void Simulation::add_receiver(const Receiver& receiver) {
  INSEM_CHECK(!receiver.name.empty() && receiver.name.find('/') == std::string::npos &&
                  receiver.name.find(',') == std::string::npos,
              "receiver name '" << receiver.name << "' must be non-empty and contain no '/' or ','");
  for (const auto& r : receivers_) {
    INSEM_CHECK(r.name != receiver.name, "duplicate receiver name '" << receiver.name << "'");
  }
  INSEM_CHECK(!receiver.quantities.empty(), "receiver '" << receiver.name << "' records no quantities");
  receivers_.push_back(receiver);
  invalidate_solver();
}

void Simulation::clear_receivers() {
  receivers_.clear();
  invalidate_solver();
}

void Simulation::set_time(const TimeConfig& time) {
  time_ = time;
  invalidate_solver();
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

solver::Solver& Simulation::solver() {
  if (!solver_) {
    setup();
    const solver::Problem problem{*forest_, *geometry_, *gll_, materials_, boundaries_,
                                  sources_, receivers_, time_};
    solver_ = std::make_unique<solver::Solver>(problem);
  }
  return *solver_;
}

void Simulation::run() { solver().run(); }

void Simulation::write_results(const std::string& path) { solver().write_results(path); }

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

  f.create_group("/materials");
  for (const auto& [region, mat] : materials_) {
    const std::string g = "/materials/" + std::to_string(region);
    f.create_group(g);
    f.set_attribute(g, "medium", to_string(mat.medium));
    f.set_attribute(g, "vp", mat.vp);
    f.set_attribute(g, "vs", mat.vs);
    f.set_attribute(g, "rho", mat.rho);
  }

  f.create_group("/boundary_conditions");
  for (const auto& [tag, type] : boundaries_) {
    f.set_attribute("/boundary_conditions", std::to_string(tag), to_string(type));
  }

  f.create_group("/sources");
  f.set_attribute("/sources", "count", static_cast<std::int64_t>(sources_.size()));
  for (std::size_t i = 0; i < sources_.size(); ++i) {
    const Source& s = sources_[i];
    const std::string g = "/sources/" + std::to_string(i);
    f.write(g + "/position", std::vector<double>{s.position[0], s.position[1]});
    f.write(g + "/components", std::vector<double>(s.components.begin(), s.components.end()));
    f.set_attribute(g, "kind", s.kind_name());
    f.set_attribute(g, "stf", s.stf.kind_name());
    f.set_attribute(g, "f0", s.stf.f0);
    f.set_attribute(g, "t0", s.stf.t0);
    f.set_attribute(g, "amplitude", s.stf.amplitude);
    if (s.stf.kind == SourceTimeFunction::Kind::Samples) {
      f.write(g + "/stf_samples", s.stf.samples);
      f.set_attribute(g, "sample_dt", s.stf.sample_dt);
    }
  }

  f.create_group("/receivers");
  f.set_attribute("/receivers", "count", static_cast<std::int64_t>(receivers_.size()));
  for (std::size_t i = 0; i < receivers_.size(); ++i) {
    const Receiver& r = receivers_[i];
    const std::string g = "/receivers/" + std::to_string(i);
    f.write(g + "/position", std::vector<double>{r.position[0], r.position[1]});
    f.set_attribute(g, "name", r.name);
    f.set_attribute(g, "quantities", join(r.quantities));
  }

  f.create_group("/time");
  f.set_attribute("/time", "dt", time_.dt);
  f.set_attribute("/time", "safety", time_.safety);
  f.set_attribute("/time", "n_steps", static_cast<std::int64_t>(time_.n_steps));
  f.set_attribute("/time", "duration", time_.duration);
  f.set_attribute("/time", "record_every", static_cast<std::int64_t>(time_.record_every));
  f.set_attribute("/time", "energy_every", static_cast<std::int64_t>(time_.energy_every));
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
  Simulation sim(std::move(m), disc);

  // Problem description groups are optional (a mesh-only case is valid).
  if (f.exists("/materials")) {
    std::set<int> regions(sim.mesh().materials().begin(), sim.mesh().materials().end());
    for (int region : regions) {
      const std::string g = "/materials/" + std::to_string(region);
      if (!f.exists(g)) continue;
      Material mat;
      mat.medium = medium_from_string(f.attribute_string(g, "medium"));
      mat.vp = f.attribute_double(g, "vp");
      mat.vs = f.attribute_double(g, "vs");
      mat.rho = f.attribute_double(g, "rho");
      sim.set_material(region, mat);
    }
  }
  if (f.exists("/boundary_conditions")) {
    std::set<int> tags;
    for (const auto& bf : sim.mesh().boundary()) tags.insert(bf.tag);
    for (int tag : tags) {
      const std::string name = std::to_string(tag);
      if (f.has_attribute("/boundary_conditions", name)) {
        sim.set_boundary(tag, boundary_type_from_string(f.attribute_string("/boundary_conditions", name)));
      }
    }
  }
  if (f.exists("/sources")) {
    const std::int64_t n = f.attribute_int("/sources", "count");
    for (std::int64_t i = 0; i < n; ++i) {
      const std::string g = "/sources/" + std::to_string(i);
      const auto pos = f.read<double>(g + "/position");
      const auto comp = f.read<double>(g + "/components");
      INSEM_CHECK(pos.size() == 2 && comp.size() == 3, "malformed source " << i);
      const std::string stf_kind = f.attribute_string(g, "stf");
      const double f0 = f.attribute_double(g, "f0");
      const double t0 = f.attribute_double(g, "t0");
      const double amp = f.attribute_double(g, "amplitude");
      SourceTimeFunction stf;
      if (stf_kind == "ricker") {
        stf = SourceTimeFunction::ricker(f0, t0, amp);
      } else if (stf_kind == "gaussian") {
        stf = SourceTimeFunction::gaussian(f0, t0, amp);
      } else if (stf_kind == "gaussian_derivative") {
        stf = SourceTimeFunction::gaussian_derivative(f0, t0, amp);
      } else if (stf_kind == "samples") {
        stf = SourceTimeFunction::sampled(f.read<double>(g + "/stf_samples"), f.attribute_double(g, "sample_dt"), amp);
      } else {
        throw Error("unknown source time function '" + stf_kind + "' in " + path);
      }
      const std::string kind = f.attribute_string(g, "kind");
      if (kind == "force") {
        sim.add_source(Source::force(pos[0], pos[1], comp[0], comp[1], stf));
      } else if (kind == "moment_tensor") {
        sim.add_source(Source::moment_tensor(pos[0], pos[1], comp[0], comp[1], comp[2], stf));
      } else if (kind == "monopole") {
        sim.add_source(Source::monopole(pos[0], pos[1], stf));
      } else {
        throw Error("unknown source kind '" + kind + "' in " + path);
      }
    }
  }
  if (f.exists("/receivers")) {
    const std::int64_t n = f.attribute_int("/receivers", "count");
    for (std::int64_t i = 0; i < n; ++i) {
      const std::string g = "/receivers/" + std::to_string(i);
      const auto pos = f.read<double>(g + "/position");
      INSEM_CHECK(pos.size() == 2, "malformed receiver " << i);
      sim.add_receiver({f.attribute_string(g, "name"), {pos[0], pos[1]}, split(f.attribute_string(g, "quantities"))});
    }
  }
  if (f.exists("/time")) {
    TimeConfig t;
    t.dt = f.attribute_double("/time", "dt");
    t.safety = f.attribute_double("/time", "safety");
    t.n_steps = static_cast<int>(f.attribute_int("/time", "n_steps"));
    t.duration = f.attribute_double("/time", "duration");
    t.record_every = static_cast<int>(f.attribute_int("/time", "record_every"));
    t.energy_every = static_cast<int>(f.attribute_int("/time", "energy_every"));
    sim.set_time(t);
  }
  return sim;
}

}  // namespace insem
