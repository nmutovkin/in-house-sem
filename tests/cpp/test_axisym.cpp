#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <functional>

#include "insem/core/error.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/solver.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem;

namespace {

using Field = std::function<std::array<double, 2>(double, double)>;

std::shared_ptr<const mesh::MacroMesh> axisym_grid(int nr, int nz, double r1 = 1.0, double z0 = -1.0,
                                                   double z1 = 1.0) {
  return test::grid(nr, nz, 0.0, r1, z0, z1, 0.0, mesh::CoordSystem::Axisymmetric);
}

// a = M^-1 f_int(u) for the given field: advance(0) initialises the solver,
// which computes the acceleration of the initial state.
std::vector<double> initial_acceleration(Simulation& sim, Medium medium, const Field& field,
                                         std::vector<double>* coords_out) {
  auto& s = sim.solver();
  const auto& d = s.domain(s.domain_index(medium));
  const std::size_t ng = d.n_global();
  std::vector<double> u(d.ncomp * ng, 0.0);
  for (std::size_t g = 0; g < ng; ++g) {
    const auto v = field(d.coords[g], d.coords[ng + g]);
    for (int c = 0; c < d.ncomp; ++c) u[c * ng + g] = v[c];
  }
  s.set_state(medium, u, std::vector<double>(u.size(), 0.0));
  s.advance(0);
  *coords_out = d.coords;
  return std::vector<double>(d.a.begin(), d.a.end());
}

// Checks a against `expected` at interior nodes (the axis included,
// the outer boundary excluded).
void check_interior(const std::vector<double>& a, const std::vector<double>& coords, int ncomp,
                    std::array<double, 2> expected, double r1, double z0, double z1, bool skip_axis_radial) {
  const std::size_t ng = coords.size() / 2;
  // Round-off scales with the largest acceleration (at the outer boundary,
  // where the omitted traction leaves the full element contributions);
  // near the axis the small r-weighted masses amplify it. Double precision
  // is exact to 1e-10 relative.
  double scale = std::max(std::abs(expected[0]), std::abs(expected[1]));
  for (double v : a) scale = std::max(scale, std::abs(v));
  const double tol = (std::is_same_v<real_t, float> ? 1e-4 : 1e-10) * scale;
  int on_axis = 0, checked = 0;
  for (std::size_t g = 0; g < ng; ++g) {
    const double r = coords[g], z = coords[ng + g];
    if (r > r1 - 1e-9 || z < z0 + 1e-9 || z > z1 - 1e-9) continue;
    for (int c = 0; c < ncomp; ++c) {
      if (c == 0 && skip_axis_radial && r == 0.0) continue;  // u_r = 0 on the axis
      CHECK_THAT(a[c * ng + g], WithinAbs(expected[c], tol));
    }
    on_axis += r == 0.0;
    ++checked;
  }
  CHECK(on_axis > 0);
  CHECK(checked > 20);
}

}  // namespace

TEST_CASE("Axis faces are rotated to face 3", "[axisym]") {
  // Unit square listed from (1, 1): the axis r = 0 is its face 1.
  const std::vector<double> nodes{1, 1, 0, 1, 0, 0, 1, 0};
  mesh::MacroMesh m(mesh::CellType::Quad4, mesh::CoordSystem::Axisymmetric, nodes, {0, 1, 2, 3}, {0},
                    {{0, 0, 7}});  // tag 7 on the top face (face 0 as given)
  CHECK(m.is_axis_face(0, 3));
  CHECK(m.boundary()[0].face == 2);  // the top face after rotation
  const auto top = m.map(0, 0.0, 1.0);
  CHECK_THAT(top[1], WithinAbs(1.0, 1e-15));
}

TEST_CASE("Axisymmetric acoustic operator is exact for quadratic pressure", "[axisym]") {
  const int order = GENERATE(3, 4, 6);
  const int level = GENERATE(0, 1);
  Simulation sim(axisym_grid(3, 4), {order, level});
  const double vp = 1.7, rho = 1.3;
  sim.set_material(0, Material::acoustic(vp, rho));
  std::vector<double> coords;
  // p = r^2 + 3 z^2: lap p = (1/r) d_r(r 2r) + 6 = 10, and p_tt = vp^2 lap p.
  const auto a = initial_acceleration(sim, Medium::Acoustic, [](double r, double z) {
    return std::array<double, 2>{r * r + 3 * z * z, 0.0};
  }, &coords);
  check_interior(a, coords, 1, {vp * vp * 10.0, 0.0}, 1.0, -1.0, 1.0, false);
}

TEST_CASE("Axisymmetric elastic operator is exact, hoop terms included", "[axisym]") {
  const int order = GENERATE(3, 4, 6);
  const int level = GENERATE(0, 1);
  const double vp = 2.0, vs = 1.1, rho = 1.4;
  const double mu = rho * vs * vs, lambda = rho * vp * vp - 2 * mu;
  struct Case {
    Field u;
    std::array<double, 2> div_sigma;
  };
  // Axisymmetric div sigma = (d_r s_rr + (s_rr - s_tt)/r + d_z s_rz, d_r s_rz + s_rz/r + d_z s_zz).
  const Case cases[] = {
      {[](double r, double) { return std::array<double, 2>{r, 0.0}; }, {0.0, 0.0}},  // uniform expansion
      {[](double r, double z) { return std::array<double, 2>{r * z, 0.0}; }, {0.0, 2 * (lambda + mu)}},
      {[](double r, double) { return std::array<double, 2>{0.0, r * r}; }, {0.0, 4 * mu}},
      {[](double, double z) { return std::array<double, 2>{0.0, z * z}; }, {0.0, 2 * (lambda + 2 * mu)}},
  };
  for (const auto& c : cases) {
    Simulation sim(axisym_grid(3, 4), {order, level});
    sim.set_material(0, Material::elastic(vp, vs, rho));
    std::vector<double> coords;
    const auto a = initial_acceleration(sim, Medium::Elastic, c.u, &coords);
    check_interior(a, coords, 2, {c.div_sigma[0] / rho, c.div_sigma[1] / rho}, 1.0, -1.0, 1.0, true);
  }
}

TEST_CASE("Axisymmetric energy is conserved", "[axisym]") {
  for (Medium medium : {Medium::Acoustic, Medium::Elastic}) {
    Discretization disc;
    disc.order = 4;
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
      u[(d.ncomp - 1) * ng + g] = std::exp(-(r * r + (z - 0.1) * (z - 0.1)) / (2 * 0.15 * 0.15));
    }
    s.set_state(medium, u, std::vector<double>(u.size(), 0.0));
    s.run();
    const double e0 = s.energy().front()[1] + s.energy().front()[2];
    for (const auto& row : s.energy()) {
      CHECK_THAT(row[1] + row[2], WithinRel(e0, std::is_same_v<real_t, float> ? 1e-4 : 1e-10));
    }
  }
}

TEST_CASE("The axis takes no boundary condition", "[axisym]") {
  Simulation sim(axisym_grid(2, 2), {3, 0});
  sim.set_material(0, Material::acoustic(1.5, 1.0));
  sim.set_boundary(4, BoundaryType::Absorbing);  // tag 4 is the left face = the axis
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("symmetry axis"));
}

TEST_CASE("Non-axisymmetric sources on the axis are rejected", "[axisym]") {
  Simulation sim(axisym_grid(2, 2), {3, 0});
  sim.set_material(0, Material::elastic(2.0, 1.0, 1.0));
  sim.add_source(Source::force(0.0, 0.0, 1.0, 0.0, SourceTimeFunction::ricker(5.0)));
  CHECK_THROWS_WITH(sim.solver(), Catch::Matchers::ContainsSubstring("radial force"));
}
