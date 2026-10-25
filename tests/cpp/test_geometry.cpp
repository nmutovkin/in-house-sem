#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "insem/basis/quadrature.hpp"
#include "insem/geometry/geometry.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem;
using geometry::GeometryKind;

namespace {

struct Setup {
  mesh::Forest forest;
  ops::DofMap dofs;
  geometry::Geometry geom;
  basis::Rule rule;
};

Setup make(std::shared_ptr<const mesh::MacroMesh> m, int level, int ngll) {
  auto forest = mesh::Forest::uniform(std::move(m), level);
  auto dofs = ops::DofMap::build(forest, ngll);
  auto rule = basis::gll(ngll);
  auto geom = geometry::compute_geometry(forest, rule.points, rule.points);
  return {std::move(forest), std::move(dofs), std::move(geom), std::move(rule)};
}

double area(const Setup& s) {
  const int n = s.rule.size();
  double a = 0.0;
  for (index_t e = 0; e < s.geom.n_elements; ++e) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        a += s.rule.weights[i] * s.rule.weights[j] * s.geom.jac[(e * n + j) * n + i];
      }
    }
  }
  return a;
}

}  // namespace

TEST_CASE("Geometry kinds are classified", "[geometry]") {
  const auto rect = make(test::grid(2, 2, 0, 3, 0, 1), 1, 5);
  for (auto k : rect.geom.kind) CHECK(k == GeometryKind::Rect);
  const auto sheared = make(test::grid(2, 2, 0, 1, 0, 1, 0.4), 1, 5);
  for (auto k : sheared.geom.kind) CHECK(k == GeometryKind::Affine);
  const auto curved = make(test::quarter_annulus(1, 2, 1.0, 2.0), 1, 5);
  for (auto k : curved.geom.kind) CHECK(k == GeometryKind::Curved);
}

TEST_CASE("Quadrature of the Jacobian gives the area", "[geometry]") {
  const int level = GENERATE(0, 1, 2);
  CHECK_THAT(area(make(test::grid(3, 2, 0, 3, -1, 1, 0.25), level, 5)), WithinRel(6.0, 1e-13));
  // Quadratic cells approximate the annulus; the error is set by the
  // geometry, not by quadrature, and shrinks with more macro-cells.
  const double exact = 0.25 * test::kPi * (4.0 - 1.0);
  CHECK_THAT(area(make(test::quarter_annulus(2, 8, 1.0, 2.0), level, 6)), WithinRel(exact, 1e-4));
}

TEST_CASE("Inverse Jacobian is the inverse", "[geometry]") {
  const auto s = make(test::quarter_annulus(2, 2, 1.0, 3.0), 1, 4);
  // det(J^-1) * det(J) = 1 at every point.
  for (std::size_t p = 0; p < s.geom.jac.size(); ++p) {
    const double inv_det = s.geom.xi_x[p] * s.geom.eta_z[p] - s.geom.xi_z[p] * s.geom.eta_x[p];
    CHECK_THAT(inv_det * s.geom.jac[p], WithinAbs(1.0, 1e-12));
  }
}

TEST_CASE("Element-local copies of shared points coincide", "[geometry]") {
  for (const auto& s : {make(test::grid(3, 3, 0, 1, 0, 1, 0.3), 2, 5),
                        make(test::quarter_annulus(3, 3, 1.0, 2.0), 2, 7)}) {
    const auto gc = geometry::global_coordinates(s.dofs, s.geom);
    CHECK(gc.max_mismatch < 1e-13);
    CHECK(gc.x.size() == static_cast<std::size_t>(s.dofs.n_global()));
  }
}
