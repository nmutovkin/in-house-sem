#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>
#include <tuple>

#include "insem/mesh/forest.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using namespace insem;
using namespace insem::mesh;

TEST_CASE("Uniform forest has 4^level leaves per tree in Morton order", "[forest]") {
  const auto m = test::grid(2, 1);
  const Forest f = Forest::uniform(m, 2);
  REQUIRE(f.n_leaves() == 2 * 16);
  // First four leaves of tree 0: the Z pattern (0,0), (1,0), (0,1), (1,1).
  CHECK((f.leaf(0).i == 0 && f.leaf(0).j == 0));
  CHECK((f.leaf(1).i == 1 && f.leaf(1).j == 0));
  CHECK((f.leaf(2).i == 0 && f.leaf(2).j == 1));
  CHECK((f.leaf(3).i == 1 && f.leaf(3).j == 1));
  CHECK((f.leaf(4).i == 2 && f.leaf(4).j == 0));
  std::set<std::tuple<int, int, int>> seen;
  for (const Leaf& l : f.leaves()) {
    CHECK(l.level == 2);
    seen.insert({l.tree, l.i, l.j});
  }
  CHECK(seen.size() == 32);
  const auto box = f.leaf_box(3);
  CHECK_THAT(box[0], WithinAbs(-0.5, 1e-15));
  CHECK_THAT(box[1], WithinAbs(0.0, 1e-15));
  CHECK_THAT(box[2], WithinAbs(-0.5, 1e-15));
  CHECK_THAT(box[3], WithinAbs(0.0, 1e-15));
}

TEST_CASE("Leaf vertices are shared across macro-cell boundaries", "[forest]") {
  const int level = GENERATE(0, 1, 2, 3);
  const int nx = 3, nz = 2;
  const Forest f = Forest::uniform(test::grid(nx, nz), level);
  const LeafVertices lv = f.leaf_vertices();
  const int s = 1 << level;
  CHECK(lv.n_vertices == (nx * s + 1) * (nz * s + 1));
}

TEST_CASE("Leaf vertices match on curved meshes with reversed edges", "[forest]") {
  const Forest f = Forest::uniform(test::quarter_annulus(2, 3), 2);
  const LeafVertices lv = f.leaf_vertices();
  CHECK(lv.n_vertices == (2 * 4 + 1) * (3 * 4 + 1));
}
