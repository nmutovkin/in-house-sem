#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <functional>

#include "insem/basis/quadrature.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/ops/element_block.hpp"
#include "insem/physics/kernels.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using namespace insem;

namespace {

using Field = std::function<std::array<double, 2>(double, double)>;

constexpr double kRho = 2.0, kLambda = 3.0, kMu = 1.5;

// Applies the internal-force operator of a homogeneous medium to the nodal
// values of `field` on grid(3, 2, 0, 1.5, 0, 1, shear) refined once, and
// checks interior nodes against mass * expected, where `expected` is
// div(sigma) (elastic) or div(grad p / rho) (acoustic) and mass the
// unit-density lumped mass. GLL quadrature is exact for these quadratic
// fields on affine elements, so this holds to round-off.
void check_operator(double shear, int ngll, bool elastic, const Field& field, std::array<double, 2> expected) {
  const double x0 = 0.0, x1 = 1.5, z0 = 0.0, z1 = 1.0;
  const auto forest = mesh::Forest::uniform(test::grid(3, 2, x0, x1, z0, z1, shear), 1);
  const auto dofs = ops::DofMap::build(forest, ngll);
  const auto rule = basis::gll(ngll);
  const auto geom = geometry::compute_geometry(forest, rule.points, rule.points);
  const std::size_t npe = static_cast<std::size_t>(ngll) * ngll;
  const std::size_t nl = npe * forest.n_leaves();
  std::vector<double> c0(nl, elastic ? kLambda : 1.0 / kRho), c1(nl, elastic ? kMu : 0.0);
  std::vector<index_t> leaves(forest.n_leaves());
  for (index_t l = 0; l < forest.n_leaves(); ++l) leaves[l] = l;
  const auto blocks = ops::build_blocks(leaves, geom, c0, c1);
  const auto kb = physics::KernelBasis::from_rule(rule.points, rule.weights);

  const int ncomp = elastic ? 2 : 1;
  const auto gc = geometry::global_coordinates(dofs, geom);
  const std::size_t ng = static_cast<std::size_t>(dofs.n_global());
  // Assembled forces of a field, and the largest element-local contribution.
  auto apply = [&](const Field& fld, std::vector<real_t>& f) {
    std::vector<real_t> u(ncomp * ng), ul(ncomp * nl), fl(ncomp * nl);
    f.assign(ncomp * ng, real_t(0));
    for (std::size_t g = 0; g < ng; ++g) {
      const auto v = fld(gc.x[g], gc.z[g]);
      for (int c = 0; c < ncomp; ++c) u[c * ng + g] = static_cast<real_t>(v[c]);
    }
    dofs.gather(u.data(), ul.data(), ncomp);
    for (const auto& b : blocks) {
      if (elastic) {
        physics::elastic_internal_forces(kb, b, ul.data(), fl.data());
      } else {
        physics::acoustic_internal_forces(kb, b, ul.data(), fl.data());
      }
    }
    dofs.assemble(fl.data(), f.data(), ncomp);
    double m = 0.0;
    for (real_t v : fl) m = std::max(m, std::abs(static_cast<double>(v)));
    return m;
  };
  std::vector<real_t> f, f_ref_field;
  apply(field, f);
  // Round-off is relative to the size of the element contributions that
  // cancel at interior nodes; a fixed quadratic field sets that scale.
  const double scale = apply([](double x, double z) { return std::array<double, 2>{x * x + z * z, x * z}; },
                             f_ref_field);

  std::vector<double> lm(nl), mass(ng);
  for (std::size_t q = 0; q < nl; ++q) {
    const std::size_t k = q % npe;
    lm[q] = rule.weights[k % ngll] * rule.weights[k / ngll] * geom.jac[q];
  }
  dofs.assemble(lm.data(), mass.data(), 1);

  const double tol = (std::is_same_v<real_t, float> ? 1e-4 : 1e-12) * scale;
  int checked = 0;
  for (std::size_t g = 0; g < ng; ++g) {
    const double x = gc.x[g] - shear * gc.z[g], z = gc.z[g];
    const double eps = 1e-9;
    if (std::abs(z - z0) < eps || std::abs(z - z1) < eps || std::abs(x - x0) < eps || std::abs(x - x1) < eps) {
      continue;  // boundary nodes carry the (omitted) traction term
    }
    for (int c = 0; c < ncomp; ++c) {
      CHECK_THAT(static_cast<double>(f[c * ng + g]), WithinAbs(expected[c] * mass[g], tol));
    }
    ++checked;
  }
  REQUIRE(checked > 0);
}

}  // namespace

TEST_CASE("Elastic operator is exact for quadratic displacements", "[kernels]") {
  const double shear = GENERATE(0.0, 0.3);  // rect and affine blocks
  const int ngll = GENERATE(3, 5, 9);
  // u = (x^2, 0): div sigma = (2 (lambda + 2 mu), 0)
  check_operator(shear, ngll, true, [](double x, double) { return std::array<double, 2>{x * x, 0.0}; },
                 {2 * (kLambda + 2 * kMu), 0.0});
  // u = (z^2, 0): div sigma = (2 mu, 0)
  check_operator(shear, ngll, true, [](double, double z) { return std::array<double, 2>{z * z, 0.0}; },
                 {2 * kMu, 0.0});
  // u = (x z, 0): div sigma = (0, lambda + mu)
  check_operator(shear, ngll, true, [](double x, double z) { return std::array<double, 2>{x * z, 0.0}; },
                 {0.0, kLambda + kMu});
  // u = (0, z^2 + x): div sigma = (0, 2 (lambda + 2 mu))
  check_operator(shear, ngll, true, [](double x, double z) { return std::array<double, 2>{0.0, z * z + x}; },
                 {0.0, 2 * (kLambda + 2 * kMu)});
}

TEST_CASE("Acoustic operator is exact for quadratic pressure", "[kernels]") {
  const double shear = GENERATE(0.0, 0.3);
  const int ngll = GENERATE(3, 5, 9);
  // p = x^2 + 3 z^2 + x z: div(grad p / rho) = 8 / rho
  check_operator(shear, ngll, false,
                 [](double x, double z) { return std::array<double, 2>{x * x + 3 * z * z + x * z, 0.0}; },
                 {8.0 / kRho, 0.0});
}

TEST_CASE("Operators annihilate rigid motions and constants", "[kernels]") {
  // Rigid rotation u = (-z, x) has zero strain; constant pressure has zero gradient.
  check_operator(0.2, 5, true, [](double x, double z) { return std::array<double, 2>{-z + 1.0, x - 2.0}; },
                 {0.0, 0.0});
  check_operator(0.2, 5, false, [](double, double) { return std::array<double, 2>{4.0, 0.0}; }, {0.0, 0.0});
}

TEST_CASE("Curved blocks agree with the affine path on affine geometry", "[kernels]") {
  // Force the curved kernel on an affine mesh by relabelling the blocks.
  const auto forest = mesh::Forest::uniform(test::grid(2, 2, 0, 1, 0, 1, 0.4), 1);
  const auto rule = basis::gll(5);
  const auto geom = geometry::compute_geometry(forest, rule.points, rule.points);
  auto curved_geom = geom;
  for (auto& k : curved_geom.kind) k = geometry::GeometryKind::Curved;
  const std::size_t nl = 25 * forest.n_leaves();
  std::vector<double> c0(nl, kLambda), c1(nl, kMu);
  std::vector<index_t> leaves(forest.n_leaves());
  for (index_t l = 0; l < forest.n_leaves(); ++l) leaves[l] = l;
  const auto affine = ops::build_blocks(leaves, geom, c0, c1);
  const auto curved = ops::build_blocks(leaves, curved_geom, c0, c1);
  REQUIRE(affine.size() == 1);
  REQUIRE(affine[0].kind == geometry::GeometryKind::Affine);
  REQUIRE(curved[0].kind == geometry::GeometryKind::Curved);
  const auto kb = physics::KernelBasis::from_rule(rule.points, rule.weights);
  std::vector<real_t> u(2 * nl), fa(2 * nl), fc(2 * nl);
  for (std::size_t i = 0; i < u.size(); ++i) u[i] = static_cast<real_t>(std::sin(0.37 * i));
  physics::elastic_internal_forces(kb, affine[0], u.data(), fa.data());
  physics::elastic_internal_forces(kb, curved[0], u.data(), fc.data());
  for (std::size_t i = 0; i < u.size(); ++i) CHECK_THAT(fa[i], WithinAbs(fc[i], 1e-3));
}
