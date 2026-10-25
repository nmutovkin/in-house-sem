// insem-run: loads a case file and runs it. The executable holds no logic of
// its own (docs/DESIGN.md §7.1); in M0 it only sets the problem up and prints
// a summary, since time stepping arrives in M1.

#include <exception>
#include <iostream>
#include <string>

#include "insem/core/config.hpp"
#include "insem/sim/simulation.hpp"

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::cerr << "usage: insem-run CASE.h5\n";
    return argc == 2 ? 0 : 2;
  }
  try {
    insem::Simulation sim = insem::Simulation::load(argv[1]);
    sim.setup();
    const auto& mesh = sim.mesh();
    std::cout << "insem " << insem::version() << "\n"
              << "  case:            " << argv[1] << "\n"
              << "  coordinates:     " << insem::mesh::to_string(mesh.coord_system()) << "\n"
              << "  macro-cells:     " << mesh.n_cells() << " ("
              << insem::mesh::to_string(mesh.cell_type()) << ")\n"
              << "  order:           " << sim.discretization().order << "\n"
              << "  elements:        " << sim.dofs().n_elements() << "\n"
              << "  global points:   " << sim.dofs().n_global() << "\n"
              << "  geometry kinds: ";
    for (const auto& [kind, count] : sim.geometry_kind_counts()) std::cout << " " << kind << "=" << count;
    std::cout << "\n";
  } catch (const std::exception& e) {
    std::cerr << "insem-run: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
