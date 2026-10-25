#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

#include "insem/basis/quadrature.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/ops/dof_map.hpp"
#include "insem/ops/element_block.hpp"
#include "insem/physics/kernels.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/solver.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem;

namespace {

// Two unit squares side by side; the right one lists its nodes starting
// from a different corner, so the shared edge is face 1 of the left cell
// and face 2 of the right cell.
std::shared_ptr<const mesh::MacroMesh> rotated_pair() {
  const std::vector<double> nodes{0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1};
  return std::make_shared<const mesh::MacroMesh>(mesh::CellType::Quad4, mesh::CoordSystem::Cartesian, nodes,
                                                 std::vector<index_t>{0, 1, 4, 3, 2, 5, 4, 1},
                                                 std::vector<int>{0, 0});
}

// Refines `times` times the leaf containing the reference point (u, v) of tree 0.
mesh::Forest refine_towards(mesh::Forest f, double u, double v, int times) {
  constexpr std::int64_t h = std::int64_t{1} << (mesh::Forest::kMaxLevel + 1);
  for (int t = 0; t < times; ++t) {
    std::vector<char> mark(f.n_leaves(), 0);
    mark[f.find_leaf(0, static_cast<std::int64_t>(u * h), static_cast<std::int64_t>(v * h))] = 1;
    f = f.refine(mark);
  }
  return f;
}

int max_jump(const mesh::Forest& f) {
  int jump = 0;
  for (index_t l = 0; l < f.n_leaves(); ++l) {
    for (int face = 0; face < 4; ++face) {
      const auto nb = f.face_neighbor(l, face);
      if (nb.leaf >= 0) jump = std::max(jump, std::abs(nb.level_diff));
    }
  }
  return jump;
}

double poly(double x, double z) { return 1.0 + x + 2 * z + x * x - x * z + 0.5 * z * z * z; }

}  // namespace

TEST_CASE("Refinement keeps Morton order and balance enforces 2:1", "[refine]") {
  const auto f0 = mesh::Forest::uniform(test::grid(2, 2), 1);
  const auto f = refine_towards(f0, 0.99, 0.99, 4);  // corner of tree 0, next to its neighbours
  CHECK(f.n_leaves() == f0.n_leaves() + 4 * 3);
  CHECK(f.max_level() == 5);
  CHECK(max_jump(f) > 1);
  const auto b = f.balance();
  CHECK(max_jump(b) <= 1);
  CHECK(b.n_leaves() > f.n_leaves());
  // Leaves of a tree still tile it: their areas add up.
  double area = 0.0;
  for (index_t l = 0; l < b.n_leaves(); ++l) {
    const auto box = b.leaf_box(l);
    area += (box[1] - box[0]) * (box[3] - box[2]);
  }
  CHECK_THAT(area, WithinRel(4.0 * 4, 1e-14));  // 4 trees of reference area 4
}

TEST_CASE("Face neighbours and face coordinates across a rotated macro-cell", "[refine]") {
  const auto macro = rotated_pair();
  const auto f = refine_towards(mesh::Forest::uniform(macro, 1), 0.9, 0.3, 1).balance();
  for (index_t l = 0; l < f.n_leaves(); ++l) {
    for (int face = 0; face < 4; ++face) {
      const auto nb = f.face_neighbor(l, face);
      if (nb.leaf < 0) continue;
      // The same physical point seen from both sides of the shared face.
      // The finer side's face lies inside the coarser side's face.
      const index_t fine = nb.level_diff <= 0 ? l : nb.leaf;
      const int fine_face = nb.level_diff <= 0 ? face : nb.face;
      const index_t coarse = nb.level_diff <= 0 ? nb.leaf : l;
      const int coarse_face = nb.level_diff <= 0 ? nb.face : face;
      if (nb.level_diff > 0 && f.face_neighbor(fine, fine_face).leaf != coarse) continue;
      for (double t : {-1.0, -0.3, 0.6, 1.0}) {
        const double tc = f.face_coordinate(fine, fine_face, t, coarse, coarse_face);
        auto ref = [](int fc, double s) {
          switch (fc) {
            case 0: return std::array<double, 2>{s, -1.0};
            case 1: return std::array<double, 2>{1.0, s};
            case 2: return std::array<double, 2>{s, 1.0};
            default: return std::array<double, 2>{-1.0, s};
          }
        };
        const auto a = ref(fine_face, t), b = ref(coarse_face, tc);
        const auto pa = geometry::leaf_map(f, fine, a[0], a[1]);
        const auto pb = geometry::leaf_map(f, coarse, b[0], b[1]);
        CHECK_THAT(pa[0], WithinAbs(pb[0], 1e-13));
        CHECK_THAT(pa[1], WithinAbs(pb[1], 1e-13));
      }
    }
  }
}

TEST_CASE("Constrained gather reproduces polynomials on refined meshes", "[refine]") {
  const int ngll = GENERATE(4, 5, 7);
  auto macro = GENERATE(as<int>{}, 0, 1);
  const auto m = macro == 0 ? rotated_pair() : test::grid(3, 2, 0, 1.5, 0, 1, 0.3);
  auto f = refine_towards(mesh::Forest::uniform(m, 1), 0.9, 0.6, 3);
  f = refine_towards(f, 0.1, 0.2, 2).balance();
  const auto dofs = ops::DofMap::build(f, ngll);
  REQUIRE(dofs.has_constraints());
  const auto rule = basis::gll(ngll);
  const auto geom = geometry::compute_geometry(f, rule.points, rule.points);
  const auto gc = geometry::global_coordinates(dofs, geom);
  CHECK(gc.max_mismatch < 1e-13);
  std::vector<double> g(dofs.n_global()), local(geom.x.size());
  for (index_t i = 0; i < dofs.n_global(); ++i) g[i] = poly(gc.x[i], gc.z[i]);
  dofs.gather(g.data(), local.data(), 1);
  double err = 0.0;
  for (std::size_t s = 0; s < local.size(); ++s) err = std::max(err, std::abs(local[s] - poly(geom.x[s], geom.z[s])));
  CHECK(err < 1e-11);

  // assemble is the transpose of the constrained gather.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> l(local.size()), lg(g.size());
  for (auto& v : l) v = dist(rng);
  dofs.assemble(l.data(), lg.data(), 1);
  double lhs = 0.0, rhs = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i) lhs += lg[i] * g[i];
  for (std::size_t i = 0; i < l.size(); ++i) rhs += l[i] * local[i];
  CHECK_THAT(lhs, WithinRel(rhs, 1e-12));

  // Lumped mass C^T M_e 1 keeps the total area and stays positive.
  std::vector<double> lm(local.size()), mass(g.size());
  for (std::size_t q = 0; q < lm.size(); ++q) {
    const std::size_t k = q % (ngll * ngll);
    lm[q] = rule.weights[k % ngll] * rule.weights[k / ngll] * geom.jac[q];
  }
  dofs.assemble(lm.data(), mass.data(), 1);
  double total = 0.0;
  for (double v : mass) {
    CHECK(v > 0.0);
    total += v;
  }
  CHECK_THAT(total, WithinRel(macro == 0 ? 2.0 : 1.5, 1e-12));
}

TEST_CASE("Elastic operator is exact on refined affine meshes", "[refine]") {
  const int ngll = 5;
  const double lambda = 3.0, mu = 1.5, shear = 0.25;
  auto f = refine_towards(mesh::Forest::uniform(test::grid(3, 2, 0, 1.5, 0, 1, shear), 1), 0.7, 0.4, 2).balance();
  const auto dofs = ops::DofMap::build(f, ngll);
  REQUIRE(dofs.has_constraints());
  const auto rule = basis::gll(ngll);
  const auto geom = geometry::compute_geometry(f, rule.points, rule.points);
  const std::size_t npe = ngll * ngll, nl = npe * f.n_leaves();
  std::vector<index_t> leaves(f.n_leaves());
  for (index_t l = 0; l < f.n_leaves(); ++l) leaves[l] = l;
  const auto blocks = ops::build_blocks(leaves, geom, std::vector<double>(nl, lambda), std::vector<double>(nl, mu));
  const auto kb = physics::KernelBasis::from_rule(rule.points, rule.weights);
  const auto gc = geometry::global_coordinates(dofs, geom);
  const std::size_t ng = dofs.n_global();
  // u = (x z, 0): div sigma = (0, lambda + mu)
  std::vector<real_t> u(2 * ng, real_t(0)), ul(2 * nl), fl(2 * nl), fg(2 * ng);
  for (std::size_t g = 0; g < ng; ++g) u[g] = static_cast<real_t>(gc.x[g] * gc.z[g]);
  dofs.gather(u.data(), ul.data(), 2);
  for (const auto& b : blocks) physics::elastic_internal_forces(kb, b, ul.data(), fl.data());
  dofs.assemble(fl.data(), fg.data(), 2);
  double scale = 0.0;
  for (real_t v : fl) scale = std::max(scale, std::abs(static_cast<double>(v)));
  std::vector<double> lm(nl), mass(ng);
  for (std::size_t q = 0; q < nl; ++q) lm[q] = rule.weights[q % npe % ngll] * rule.weights[q % npe / ngll] * geom.jac[q];
  dofs.assemble(lm.data(), mass.data(), 1);
  const double tol = (std::is_same_v<real_t, float> ? 1e-4 : 1e-11) * scale;
  int checked = 0;
  for (std::size_t g = 0; g < ng; ++g) {
    const double x = gc.x[g] - shear * gc.z[g], z = gc.z[g];
    if (z < 1e-9 || z > 1 - 1e-9 || x < 1e-9 || x > 1.5 - 1e-9) continue;
    CHECK_THAT(static_cast<double>(fg[g]), WithinAbs(0.0, tol));
    CHECK_THAT(static_cast<double>(fg[ng + g]), WithinAbs((lambda + mu) * mass[g], tol));
    ++checked;
  }
  CHECK(checked > 100);
}

TEST_CASE("Energy is conserved on a locally refined mesh", "[refine]") {
  Discretization disc;
  disc.order = 4;
  disc.uniform_level = 1;
  disc.regions = {{0.3, 0.6, 0.4, 0.7, 3}};
  Simulation sim(test::grid(3, 3), disc);
  sim.set_material(0, Material::elastic(2.0, 1.1, 1.3));
  TimeConfig t;
  t.n_steps = 300;
  t.energy_every = 30;
  sim.set_time(t);
  auto& s = sim.solver();
  REQUIRE(sim.dofs().has_constraints());
  REQUIRE(sim.forest().max_level() == 3);
  const auto& d = s.domain(0);
  const std::size_t ng = d.n_global();
  std::vector<double> u(2 * ng, 0.0);
  for (std::size_t g = 0; g < ng; ++g) {
    const double x = d.coords[g], z = d.coords[ng + g];
    u[g] = std::exp(-((x - 0.45) * (x - 0.45) + (z - 0.55) * (z - 0.55)) / (2 * 0.1 * 0.1));
  }
  s.set_state(Medium::Elastic, u, std::vector<double>(u.size(), 0.0));
  s.run();
  const double e0 = s.energy().front()[1] + s.energy().front()[2];
  for (const auto& row : s.energy()) {
    CHECK_THAT(row[1] + row[2], WithinRel(e0, std::is_same_v<real_t, float> ? 1e-4 : 1e-10));
  }
}

TEST_CASE("Wavelength refinement resolves slow regions and matches fluid-solid levels", "[refine]") {
  // Slow fluid above a fast solid: the fluid needs finer leaves, and the
  // interface faces must still be conforming.
  auto g = test::grid(4, 4);
  std::vector<int> mat(16);
  for (int c = 0; c < 16; ++c) mat[c] = c < 8 ? 1 : 0;  // bottom half solid (region 1)
  auto m = std::make_shared<const mesh::MacroMesh>(g->cell_type(), g->coord_system(), g->nodes(), g->cells(), mat,
                                                   g->boundary());
  Discretization disc;
  disc.order = 4;
  disc.f_max = 4.0;
  Simulation sim(m, disc);
  sim.set_material(0, Material::acoustic(1.0, 1.0));
  sim.set_material(1, Material::elastic(6.0, 3.5, 2.0));
  sim.setup();
  const auto& f = sim.forest();
  for (index_t l = 0; l < f.n_leaves(); ++l) {
    const auto box = f.leaf_box(l);
    const double h = 0.25 * 0.5 * (box[1] - box[0]);  // grid cells are 0.25 wide
    const bool fluid = m->material(f.leaf(l).tree) == 0;
    const double v_min = fluid ? 1.0 : 3.5;
    CHECK(h <= 4 * (v_min / 4.0) / 5.0 + 1e-12);
  }
  TimeConfig t;
  t.n_steps = 50;
  sim.set_time(t);
  sim.add_source(Source::monopole(0.5, 0.8, SourceTimeFunction::ricker(4.0)));
  CHECK_NOTHROW(sim.run());  // DofMap would reject unmatched levels at the interface
}
