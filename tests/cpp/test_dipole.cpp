// Azimuthal order 1 (dipole) fields in axisymmetric models (docs/DESIGN.md §2.5).

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <functional>

#include "insem/core/error.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/solver.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem;

namespace {

using Vec3 = std::array<double, 3>;
using Field3 = std::function<Vec3(double, double)>;

const double kSqrt2 = std::sqrt(2.0);

std::shared_ptr<const mesh::MacroMesh> axisym_grid(int nr, int nz, double r1 = 1.0, double z0 = -1.0,
                                                   double z1 = 1.0) {
  return test::grid(nr, nz, 0.0, r1, z0, z1, 0.0, mesh::CoordSystem::Axisymmetric);
}

Discretization dipole_disc(int order, int level = 0) {
  Discretization d;
  d.order = order;
  d.uniform_level = level;
  d.azimuthal_order = 1;
  return d;
}

// Checks M^-1 f_int(u) of the field u (in the domain's components) against
// `expected` at the interior nodes (the axis included, its fixed components
// excluded; the outer boundary excluded, where the omitted traction leaves
// the full element contributions).
void check_operator(Simulation& sim, Medium medium, const Field3& u, const Field3& expected) {
  auto& s = sim.solver();
  const auto& d = s.domain(s.domain_index(medium));
  const std::size_t ng = d.n_global();
  std::vector<double> x(d.ncomp * ng, 0.0);
  for (std::size_t g = 0; g < ng; ++g) {
    const Vec3 v = u(d.coords[g], d.coords[ng + g]);
    for (int c = 0; c < d.ncomp; ++c) x[c * ng + g] = v[c];
  }
  s.set_state(medium, x, std::vector<double>(x.size(), 0.0));
  s.advance(0);  // computes the acceleration of the initial state
  double scale = 0.0;
  for (real_t v : d.a) scale = std::max(scale, std::abs(static_cast<double>(v)));
  const double tol = (std::is_same_v<real_t, float> ? 1e-4 : 1e-10) * scale;
  int on_axis = 0, checked = 0;
  for (std::size_t g = 0; g < ng; ++g) {
    const double r = d.coords[g], z = d.coords[ng + g];
    if (r > 1.0 - 1e-9 || z < -1.0 + 1e-9 || z > 1.0 - 1e-9) continue;
    const Vec3 want = expected(r, z);
    for (int c = 0; c < d.ncomp; ++c) {
      const bool fixed = r == 0.0 && std::find(d.axis_components.begin(), d.axis_components.end(), c) !=
                                         d.axis_components.end();
      if (fixed) {
        CHECK(d.a[c * ng + g] == 0);
        continue;
      }
      CHECK_THAT(d.a[c * ng + g], WithinAbs(want[c], tol));
    }
    on_axis += r == 0.0;
    ++checked;
  }
  CHECK(on_axis > 0);
  CHECK(checked > 20);
}

double gauss(double r, double z) { return std::exp(-(r * r + (z - 0.1) * (z - 0.1)) / (2 * 0.15 * 0.15)); }

}  // namespace

TEST_CASE("Azimuthal order 1 acoustic operator is exact for cubic pressure", "[dipole]") {
  const int order = GENERATE(3, 4, 6);
  const int level = GENERATE(0, 1);
  const double vp = 1.7, rho = 1.3;
  // p = P(r, z) cos(theta); P = r z^2 is p = x z^2 with p_tt = vp^2 lap p =
  // 2 vp^2 x, and P = r^3 is p = x (x^2 + y^2) with lap p = 8 x.
  struct Case {
    Field3 p, p_tt;
  };
  const Case cases[] = {
      {[](double r, double z) { return Vec3{r * z * z, 0, 0}; },
       [&](double r, double) { return Vec3{2 * vp * vp * r, 0, 0}; }},
      {[](double r, double) { return Vec3{r * r * r, 0, 0}; },
       [&](double r, double) { return Vec3{8 * vp * vp * r, 0, 0}; }},
  };
  for (const auto& c : cases) {
    Simulation sim(axisym_grid(3, 4), dipole_disc(order, level));
    sim.set_material(0, Material::acoustic(vp, rho));
    check_operator(sim, Medium::Acoustic, c.p, c.p_tt);
  }
}

TEST_CASE("Azimuthal order 1 elastic operator is exact, 1/r terms included", "[dipole]") {
  const int order = GENERATE(3, 4, 6);
  const int level = GENERATE(0, 1);
  const double vp = 2.0, vs = 1.1, rho = 1.4;
  const double mu = rho * vs * vs, lambda = rho * vp * vp - 2 * mu;
  // Cartesian fields u with div sigma = mu lap u + (lambda + mu) grad div u,
  // in mode amplitudes (U+, U-, U_z): a field F e_x has F+ = 0, F- = sqrt2 F.
  struct Case {
    Field3 u, div_sigma;
  };
  const Case cases[] = {
      // u = (z^2, 0, 0): U_r = z^2, U_theta = -z^2
      {[](double, double z) { return Vec3{0, kSqrt2 * z * z, 0}; },
       [&](double, double) { return Vec3{0, kSqrt2 * 2 * mu, 0}; }},
      // u = (0, 0, x z): U_z = r z
      {[](double r, double z) { return Vec3{0, 0, r * z}; },
       [&](double, double) { return Vec3{0, kSqrt2 * (lambda + mu), 0}; }},
      // u = (x^2 + y^2) e_x
      {[](double r, double) { return Vec3{0, kSqrt2 * r * r, 0}; },
       [&](double, double) { return Vec3{0, kSqrt2 * (2 * lambda + 6 * mu), 0}; }},
      // u = (x^2, x y, 0): U_r = r^2, U_theta = 0, so U+ = U- = r^2 / sqrt2
      {[](double r, double) { return Vec3{r * r / kSqrt2, r * r / kSqrt2, 0}; },
       [&](double, double) { return Vec3{0, kSqrt2 * (3 * lambda + 5 * mu), 0}; }},
      // u = (0, 0, x (x^2 + y^2)): U_z = r^3, div sigma = (0, 0, 8 mu x)
      {[](double r, double) { return Vec3{0, 0, r * r * r}; },
       [&](double r, double) { return Vec3{0, 0, 8 * mu * r}; }},
  };
  for (const auto& c : cases) {
    Simulation sim(axisym_grid(3, 4), dipole_disc(order, level));
    sim.set_material(0, Material::elastic(vp, vs, rho));
    check_operator(sim, Medium::Elastic, c.u, [&](double r, double z) {
      Vec3 v = c.div_sigma(r, z);
      for (double& x : v) x /= rho;
      return v;
    });
  }
}

TEST_CASE("Azimuthal order 1 energy is conserved", "[dipole]") {
  for (Medium medium : {Medium::Acoustic, Medium::Elastic}) {
    Discretization disc = dipole_disc(4);
    disc.regions = {{0.0, 0.3, -0.2, 0.2, 1}};  // hanging faces next to the axis
    Simulation sim(axisym_grid(4, 6), disc);
    sim.set_material(0, medium == Medium::Elastic ? Material::elastic(2.0, 1.1, 1.3) : Material::acoustic(1.5, 1.0));
    TimeConfig t;
    t.n_steps = 300;
    t.energy_every = 30;
    sim.set_time(t);
    auto& s = sim.solver();
    REQUIRE(sim.dofs().has_constraints());
    const auto& d = s.domain(0);
    const std::size_t ng = d.n_global();
    std::vector<double> u(d.ncomp * ng, 0.0);
    for (std::size_t g = 0; g < ng; ++g) {
      const double r = d.coords[g], z = d.coords[ng + g];
      for (int c = 0; c < d.ncomp; ++c) u[c * ng + g] = (c == 1 ? 1.0 : r) * gauss(r, z);
    }
    s.set_state(medium, u, std::vector<double>(u.size(), 0.0));
    s.run();
    const double e0 = s.energy().front()[1] + s.energy().front()[2];
    REQUIRE(e0 > 0.0);
    for (const auto& row : s.energy()) {
      CHECK_THAT(row[1] + row[2], WithinRel(e0, std::is_same_v<real_t, float> ? 1e-4 : 1e-10));
    }
  }
}

TEST_CASE("Azimuthal order 1 Stacey boundaries absorb", "[dipole]") {
  for (Medium medium : {Medium::Acoustic, Medium::Elastic}) {
    Simulation sim(axisym_grid(4, 6), dipole_disc(4));
    sim.set_material(0, medium == Medium::Elastic ? Material::elastic(2.0, 1.1, 1.3) : Material::acoustic(1.5, 1.0));
    for (int tag : {1, 2, 3}) sim.set_boundary(tag, BoundaryType::Absorbing);
    TimeConfig t;
    t.duration = 3.0;
    t.energy_every = 20;
    sim.set_time(t);
    auto& s = sim.solver();
    const auto& d = s.domain(0);
    const std::size_t ng = d.n_global();
    std::vector<double> u(d.ncomp * ng, 0.0);
    for (std::size_t g = 0; g < ng; ++g) {
      const double r = d.coords[g], z = d.coords[ng + g];
      for (int c = 0; c < d.ncomp; ++c) u[c * ng + g] = (c == 1 ? 1.0 : r) * gauss(r, z);
    }
    s.set_state(medium, u, std::vector<double>(u.size(), 0.0));
    s.run();
    const auto& e = s.energy();
    const double e0 = e.front()[1] + e.front()[2];
    for (std::size_t i = 1; i < e.size(); ++i) {
      CHECK(e[i][1] + e[i][2] <= (e[i - 1][1] + e[i - 1][2]) * (1 + 1e-6));
    }
    CHECK(e.back()[1] + e.back()[2] < 0.05 * e0);
  }
}

TEST_CASE("Dipole sources drive a coupled borehole model stably", "[dipole][coupling]") {
  // Fluid column r < 0.25 inside a solid, as in a borehole.
  auto m = axisym_grid(4, 8);
  std::vector<int> material(m->n_cells());
  for (index_t c = 0; c < m->n_cells(); ++c) material[c] = c % 4 == 0 ? 0 : 1;
  auto mixed = std::make_shared<const mesh::MacroMesh>(m->cell_type(), m->coord_system(), m->nodes(), m->cells(),
                                                       material, m->boundary());
  Simulation sim(mixed, dipole_disc(4));
  sim.set_material(0, Material::acoustic(1.5, 1.0));
  sim.set_material(1, Material::elastic(3.0, 1.2, 2.0));
  for (int tag : {1, 2, 3}) sim.set_boundary(tag, BoundaryType::Absorbing);
  sim.add_source(Source::dipole(0.0, 0.0, 1.0, 0.0, SourceTimeFunction::ricker(2.0)));
  sim.add_receiver({"fluid", {0.1, 0.5}, {"pressure"}, ""});
  sim.add_receiver({"solid", {0.5, 0.5}, {"velocity"}, ""});
  TimeConfig t;
  t.duration = 4.0;
  sim.set_time(t);
  auto& s = sim.solver();
  REQUIRE(s.n_domains() == 2);
  s.run();
  const auto& traces = s.traces();
  REQUIRE(traces[1].ncomp == 3);
  double pmax = 0.0, vmax[3] = {0, 0, 0};
  for (double p : traces[0].data) {
    REQUIRE(std::isfinite(p));
    pmax = std::max(pmax, std::abs(p));
  }
  for (std::size_t i = 0; i < traces[1].data.size(); ++i) {
    REQUIRE(std::isfinite(traces[1].data[i]));
    vmax[i % 3] = std::max(vmax[i % 3], std::abs(traces[1].data[i]));
  }
  CHECK(pmax > 0.0);
  for (double v : vmax) CHECK(v > 0.0);  // u_r, u_theta and u_z all move
}

TEST_CASE("Azimuthal order 1 rejects axisymmetric sources and Cartesian meshes", "[dipole]") {
  const auto stf = SourceTimeFunction::ricker(5.0);
  CHECK_THROWS_WITH(Simulation(test::grid(2, 2), dipole_disc(3)), ContainsSubstring("axisymmetric mesh"));
  {
    Simulation sim(axisym_grid(2, 2), dipole_disc(3));
    sim.set_material(0, Material::acoustic(1.5, 1.0));
    sim.add_source(Source::monopole(0.0, 0.0, stf));
    CHECK_THROWS_WITH(sim.solver(), ContainsSubstring("force and dipole sources only"));
  }
  {
    Simulation sim(axisym_grid(2, 2), dipole_disc(3));
    sim.set_material(0, Material::elastic(2.0, 1.0, 1.0));
    sim.add_source(Source::force(0.0, 0.0, 0.0, 1.0, stf));
    CHECK_THROWS_WITH(sim.solver(), ContainsSubstring("vertical force or dipole"));
  }
  {
    // ... and order 0 rejects a horizontal dipole on the axis.
    Simulation sim(axisym_grid(2, 2), {3, 0});
    sim.set_material(0, Material::acoustic(1.5, 1.0));
    sim.add_source(Source::dipole(0.0, 0.0, 1.0, 0.0, stf));
    CHECK_THROWS_WITH(sim.solver(), ContainsSubstring("radial force or dipole"));
  }
}

TEST_CASE("Azimuthal order and dipole sources round-trip through the case file", "[dipole]") {
  Simulation sim(axisym_grid(2, 2), dipole_disc(3));
  sim.set_material(0, Material::acoustic(1.5, 1.0));
  sim.add_source(Source::dipole(0.0, 0.25, 2.0, 0.0, SourceTimeFunction::ricker(5.0)));
  const auto path = std::filesystem::temp_directory_path() / "insem_dipole_case.h5";
  sim.save(path.string());
  const Simulation back = Simulation::load(path.string());
  std::filesystem::remove(path);
  CHECK(back.discretization().azimuthal_order == 1);
  REQUIRE(back.sources().size() == 1);
  CHECK(back.sources()[0].kind == Source::Kind::Dipole);
  CHECK(back.sources()[0].medium() == Medium::Acoustic);
  CHECK(back.sources()[0].components[0] == 2.0);
}
