#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

#include "insem/ops/dof_map.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using namespace insem;

TEST_CASE("Global point count of a structured mesh", "[dofs]") {
  const int ngll = GENERATE(2, 3, 5, 8);
  const int level = GENERATE(0, 1, 2);
  const int nx = 3, nz = 2;
  const auto forest = mesh::Forest::uniform(test::grid(nx, nz), level);
  const auto dofs = ops::DofMap::build(forest, ngll);
  const int s = 1 << level;
  const int n = ngll - 1;
  CHECK(dofs.n_elements() == nx * nz * s * s);
  CHECK(dofs.n_global() == (nx * s * n + 1) * (nz * s * n + 1));
}

TEST_CASE("Multiplicity: 1 inside, 2 on edges, up to 4 at vertices", "[dofs]") {
  const auto dofs = ops::DofMap::build(mesh::Forest::uniform(test::grid(2, 2), 0), 4);
  const auto mult = dofs.multiplicity();
  int count[5] = {0, 0, 0, 0, 0};
  for (index_t m : mult) {
    REQUIRE((m >= 1 && m <= 4));
    ++count[m];
  }
  CHECK(count[4] == 1);           // centre vertex
  CHECK(count[2] == 4 * 2 + 4);   // 4 interior edges x 2 points + 4 boundary vertices
  CHECK(count[1] == dofs.n_global() - count[4] - count[2]);
}

TEST_CASE("assemble is the exact transpose of gather", "[dofs]") {
  const auto dofs = ops::DofMap::build(mesh::Forest::uniform(test::quarter_annulus(2, 2), 1), 5);
  const int ncomp = 2;
  const std::size_t ng = static_cast<std::size_t>(dofs.n_global()) * ncomp;
  const std::size_t nl = static_cast<std::size_t>(dofs.n_elements()) * ncomp * dofs.points_per_element();
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> g(ng), l(nl), gl(nl), lg(ng);
  for (auto& v : g) v = dist(rng);
  for (auto& v : l) v = dist(rng);
  dofs.gather(g.data(), gl.data(), ncomp);
  dofs.assemble(l.data(), lg.data(), ncomp);
  double lhs = 0.0, rhs = 0.0;  // <assemble(l), g> == <l, gather(g)>
  for (std::size_t i = 0; i < ng; ++i) lhs += lg[i] * g[i];
  for (std::size_t i = 0; i < nl; ++i) rhs += l[i] * gl[i];
  CHECK_THAT(lhs, WithinAbs(rhs, 1e-10));
}

TEST_CASE("assemble(gather(1)) is the multiplicity", "[dofs]") {
  const auto dofs = ops::DofMap::build(mesh::Forest::uniform(test::grid(3, 3), 1), 3);
  std::vector<float> ones(dofs.n_global(), 1.0f), local(static_cast<std::size_t>(dofs.n_elements()) * 9);
  std::vector<float> back(dofs.n_global());
  dofs.gather(ones.data(), local.data(), 1);
  dofs.assemble(local.data(), back.data(), 1);
  const auto mult = dofs.multiplicity();
  for (index_t gi = 0; gi < dofs.n_global(); ++gi) CHECK(back[gi] == static_cast<float>(mult[gi]));
}

TEST_CASE("assemble is bitwise deterministic", "[dofs]") {
  const auto dofs = ops::DofMap::build(mesh::Forest::uniform(test::grid(4, 4), 2), 5);
  const std::size_t nl = static_cast<std::size_t>(dofs.n_elements()) * dofs.points_per_element();
  std::vector<float> l(nl);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : l) v = dist(rng);
  std::vector<float> a(dofs.n_global()), b(dofs.n_global());
  dofs.assemble(l.data(), a.data(), 1);
  for (int rep = 0; rep < 5; ++rep) {
    dofs.assemble(l.data(), b.data(), 1);
    REQUIRE(a == b);
  }
}
