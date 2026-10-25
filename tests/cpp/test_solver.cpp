#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>

#include "insem/basis/quadrature.hpp"
#include "insem/core/error.hpp"
#include "insem/io/hdf5.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/locate.hpp"
#include "insem/solver/solver.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem;

namespace {

double total_energy(const std::array<double, 3>& e) { return e[1] + e[2]; }

// Smooth bump centred in the unit square.
double bump(double x, double z) {
  const double r2 = (x - 0.45) * (x - 0.45) + (z - 0.55) * (z - 0.55);
  return std::exp(-r2 / (2 * 0.08 * 0.08));
}

std::vector<double> sample(const solver::Domain& d, int ncomp, double (*f)(double, double)) {
  const std::size_t ng = static_cast<std::size_t>(d.n_global());
  std::vector<double> out(ncomp * ng, 0.0);
  for (std::size_t g = 0; g < ng; ++g) out[g] = f(d.coords[g], d.coords[ng + g]);
  return out;
}

}  // namespace

TEST_CASE("Point location inverts curved mappings", "[solver]") {
  const auto forest = mesh::Forest::uniform(test::quarter_annulus(2, 3, 1.0, 2.0), 1);
  const auto rule = basis::gll(5);
  const auto geom = geometry::compute_geometry(forest, rule.points, rule.points);
  std::vector<index_t> all(forest.n_leaves());
  for (index_t l = 0; l < forest.n_leaves(); ++l) all[l] = l;
  for (const std::array<double, 2> x : {std::array<double, 2>{1.3, 0.4}, {0.2, 1.7}, {1.0, 1.0}}) {
    const auto loc = solver::locate(forest, geom, x, all);
    REQUIRE(loc);
    const auto y = geometry::leaf_map(forest, loc->leaf, loc->xi, loc->eta);
    CHECK_THAT(y[0], WithinAbs(x[0], 1e-12));
    CHECK_THAT(y[1], WithinAbs(x[1], 1e-12));
  }
  CHECK_FALSE(solver::locate(forest, geom, {0.1, 0.1}, all));  // inside the hole
}

TEST_CASE("Elastic energy is conserved in a closed box", "[solver]") {
  Simulation sim(test::grid(6, 6), {4, 0});
  sim.set_material(0, Material::elastic(2.0, 1.1, 1.3));
  TimeConfig t;
  t.n_steps = 400;
  t.energy_every = 20;
  sim.set_time(t);
  auto& s = sim.solver();
  const auto& d = s.domain(s.domain_index(Medium::Elastic));
  auto u = sample(d, 2, bump);
  s.set_state(Medium::Elastic, u, std::vector<double>(u.size(), 0.0));
  s.run();
  const auto& e = s.energy();
  REQUIRE(e.size() == 21);
  const double e0 = total_energy(e.front());
  REQUIRE(e0 > 0.0);
  const double tol = std::is_same_v<real_t, float> ? 1e-4 : 1e-10;
  for (const auto& row : e) CHECK_THAT(total_energy(row), WithinRel(e0, tol));
  // Energy actually moved between kinetic and potential.
  double max_kinetic = 0.0;
  for (const auto& row : e) max_kinetic = std::max(max_kinetic, row[1]);
  CHECK(max_kinetic > 0.2 * e0);
}

TEST_CASE("Acoustic energy is conserved with rigid walls and decays with absorbing ones", "[solver]") {
  for (bool absorbing : {false, true}) {
    Simulation sim(test::grid(8, 8), {4, 0});
    sim.set_material(0, Material::acoustic(1.5, 1.0));
    if (absorbing) {
      for (int tag : {1, 2, 3, 4}) sim.set_boundary(tag, BoundaryType::Absorbing);
    }
    TimeConfig t;
    t.duration = 3.0;  // two crossings of the box
    t.energy_every = 10;
    sim.set_time(t);
    auto& s = sim.solver();
    const auto& d = s.domain(0);
    auto p = sample(d, 1, bump);
    s.set_state(Medium::Acoustic, p, std::vector<double>(p.size(), 0.0));
    s.run();
    const double e0 = total_energy(s.energy().front());
    const double e1 = total_energy(s.energy().back());
    if (absorbing) {
      CHECK(e1 < 0.01 * e0);
    } else {
      CHECK_THAT(e1, WithinRel(e0, std::is_same_v<real_t, float> ? 1e-4 : 1e-10));
    }
  }
}

TEST_CASE("Rigid elastic boundaries stay at rest", "[solver]") {
  Simulation sim(test::grid(4, 4), {3, 0});
  sim.set_material(0, Material::elastic(2.0, 1.0, 1.0));
  for (int tag : {1, 2, 3, 4}) sim.set_boundary(tag, BoundaryType::Rigid);
  sim.add_source(Source::force(0.5, 0.5, 1.0, 1.0, SourceTimeFunction::ricker(2.0)));
  TimeConfig t;
  t.n_steps = 100;
  sim.set_time(t);
  auto& s = sim.solver();
  s.run();
  const auto& d = s.domain(0);
  REQUIRE(!d.dirichlet_points.empty());
  CHECK(d.absorbing_points.empty());
  const std::size_t ng = static_cast<std::size_t>(d.n_global());
  double interior = 0.0;
  for (index_t g : d.dirichlet_points) {
    CHECK(d.u[g] == 0);
    CHECK(d.u[ng + g] == 0);
  }
  for (std::size_t g = 0; g < ng; ++g) interior = std::max(interior, std::abs(static_cast<double>(d.u[g])));
  CHECK(interior > 0.0);
}

TEST_CASE("Stability limit is respected and enforced", "[solver]") {
  Simulation sim(test::grid(4, 4), {4, 0});
  sim.set_material(0, Material::elastic(3.0, 1.5, 1.0));
  const double stable = sim.solver().stable_dt();
  // CFL-type sanity: the limit scales with the smallest GLL spacing / vp.
  const double h = 0.25 * (1.0 - std::sqrt(3.0 / 7.0)) / 2.0;
  CHECK(stable > 0.2 * h / 3.0);
  CHECK(stable < 3.0 * h / 3.0);
  TimeConfig t;
  t.dt = 1.5 * stable;
  t.n_steps = 1;
  sim.set_time(t);
  CHECK_THROWS_AS(sim.solver(), insem::Error);
}

TEST_CASE("Fluid-solid interfaces are rejected until M2", "[solver]") {
  auto m = test::grid(2, 1);
  auto mixed = std::make_shared<const mesh::MacroMesh>(m->cell_type(), m->coord_system(), m->nodes(), m->cells(),
                                                       std::vector<int>{0, 1}, m->boundary());
  Simulation sim(mixed, {3, 0});
  sim.set_material(0, Material::acoustic(1.5, 1.0));
  sim.set_material(1, Material::elastic(3.0, 1.5, 2.0));
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("fluid-solid"));
}

TEST_CASE("Sources and receivers outside their medium fail clearly", "[solver]") {
  Simulation sim(test::grid(2, 2), {3, 0});
  sim.set_material(0, Material::elastic(2.0, 1.0, 1.0));
  sim.add_source(Source::monopole(0.5, 0.5, SourceTimeFunction::ricker(5.0)));
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("acoustic"));
  sim.clear_sources();
  sim.add_receiver({"r1", {3.0, 0.5}, {"velocity"}});
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("outside"));
  sim.clear_receivers();
  sim.add_receiver({"r1", {0.5, 0.5}, {"pressure"}});
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("cannot be recorded"));
}

TEST_CASE("Runs are deterministic and results round-trip", "[solver]") {
  const auto path = (std::filesystem::temp_directory_path() / "insem_test_results.h5").string();
  const auto case_path = (std::filesystem::temp_directory_path() / "insem_test_run_case.h5").string();
  auto build = [] {
    Simulation sim(test::grid(5, 4, 0, 1, 0, 0.8, 0.1), {4, 1});
    sim.set_material(0, Material::elastic(2.0, 1.0, 1.0));
    sim.set_boundary(1, BoundaryType::Absorbing);
    sim.set_boundary(2, BoundaryType::Absorbing);
    sim.add_source(Source::moment_tensor(0.42, 0.37, 1.0, 0.5, 0.2, SourceTimeFunction::ricker(8.0)));
    sim.add_receiver({"a", {0.7, 0.6}, {"displacement", "velocity"}});
    TimeConfig t;
    t.n_steps = 150;
    t.record_every = 3;
    sim.set_time(t);
    return sim;
  };
  Simulation a = build();
  a.save(case_path);
  Simulation b = Simulation::load(case_path);
  a.run();
  b.run();
  const auto& ta = a.solver().traces();
  const auto& tb = b.solver().traces();
  REQUIRE(ta.size() == 2);
  CHECK(ta[0].data == tb[0].data);  // bitwise identical
  CHECK(ta[0].data.size() == 2 * 51);
  a.write_results(path);
  io::H5File f(path, io::H5File::Mode::Read);
  std::vector<std::size_t> dims;
  CHECK(f.read<double>("/receivers/a/velocity", &dims) == ta[1].data);
  CHECK(dims == std::vector<std::size_t>{51, 2});
  CHECK(f.read<double>("/time").size() == 51);
  CHECK(f.attribute_string("/receivers/a", "medium") == "elastic");
  std::filesystem::remove(path);
  std::filesystem::remove(case_path);
}
