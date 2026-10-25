#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "insem/core/error.hpp"
#include "insem/sim/simulation.hpp"
#include "test_helpers.hpp"

using namespace insem;

TEST_CASE("Simulation setup derives forest, dofs and geometry", "[sim]") {
  Simulation sim(test::grid(4, 2, 0, 2, 0, 1), {4, 1});
  CHECK_FALSE(sim.is_setup());
  CHECK_THROWS_AS(sim.dofs(), insem::Error);
  sim.setup();
  CHECK(sim.dofs().n_elements() == 32);
  CHECK(sim.dofs().n_global() == (4 * 2 * 4 + 1) * (2 * 2 * 4 + 1));
  CHECK(sim.geometry_kind_counts().at("rect") == 32);
}

TEST_CASE("Case file round-trips the problem description", "[sim]") {
  const auto path = (std::filesystem::temp_directory_path() / "insem_test_case.h5").string();
  Simulation sim(test::quarter_annulus(2, 3, 0.5, 1.5), {6, 2});
  sim.save(path);
  Simulation loaded = Simulation::load(path);
  CHECK(loaded.discretization().order == 6);
  CHECK(loaded.discretization().uniform_level == 2);
  CHECK(loaded.mesh().cell_type() == mesh::CellType::Quad9);
  CHECK(loaded.mesh().nodes() == sim.mesh().nodes());
  CHECK(loaded.mesh().cells() == sim.mesh().cells());
  loaded.setup();
  sim.setup();
  CHECK(loaded.dofs().local_to_global() == sim.dofs().local_to_global());
  CHECK(loaded.geometry().jac == sim.geometry().jac);
  std::filesystem::remove(path);
}

TEST_CASE("Case file keeps boundary tags and coordinate system", "[sim]") {
  const auto path = (std::filesystem::temp_directory_path() / "insem_test_case_axi.h5").string();
  Simulation sim(test::grid(2, 3, 0, 1, 0, 1, 0.0, mesh::CoordSystem::Axisymmetric), {3, 0});
  sim.save(path);
  const Simulation loaded = Simulation::load(path);
  CHECK(loaded.mesh().coord_system() == mesh::CoordSystem::Axisymmetric);
  REQUIRE(loaded.mesh().boundary().size() == sim.mesh().boundary().size());
  for (std::size_t i = 0; i < sim.mesh().boundary().size(); ++i) {
    CHECK(loaded.mesh().boundary()[i].cell == sim.mesh().boundary()[i].cell);
    CHECK(loaded.mesh().boundary()[i].face == sim.mesh().boundary()[i].face);
    CHECK(loaded.mesh().boundary()[i].tag == sim.mesh().boundary()[i].tag);
  }
  std::filesystem::remove(path);
}

TEST_CASE("Loading a non-case file fails clearly", "[sim]") {
  CHECK_THROWS_AS(Simulation::load("/nonexistent.h5"), insem::Error);
}
