// insem-run: loads a case file, runs it and writes the results. The
// executable holds no logic of its own (docs/DESIGN.md §7.1).

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#include "insem/core/config.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/solver.hpp"

namespace {

std::string default_results_path(const std::string& case_path) {
  const auto dot = case_path.rfind(".h5");
  const std::string stem = dot == std::string::npos ? case_path : case_path.substr(0, dot);
  return stem + "_results.h5";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    std::cerr << "usage: insem-run CASE.h5 [RESULTS.h5]\n";
    return argc >= 2 ? 0 : 2;
  }
  const std::string case_path = argv[1];
  const std::string results_path = argc == 3 ? argv[2] : default_results_path(case_path);
  try {
    insem::Simulation sim = insem::Simulation::load(case_path);
    sim.setup();
    const auto& mesh = sim.mesh();
    std::cout << "insem " << insem::version() << "\n"
              << "  case:            " << case_path << "\n"
              << "  coordinates:     " << insem::mesh::to_string(mesh.coord_system()) << "\n"
              << "  macro-cells:     " << mesh.n_cells() << " ("
              << insem::mesh::to_string(mesh.cell_type()) << ")\n"
              << "  order:           " << sim.discretization().order << "\n"
              << "  elements:        " << sim.dofs().n_elements() << "\n"
              << "  geometry kinds: ";
    for (const auto& [kind, count] : sim.geometry_kind_counts()) std::cout << " " << kind << "=" << count;
    std::cout << "\n";

    insem::solver::Solver& solver = sim.solver();
    for (std::size_t i = 0; i < solver.n_domains(); ++i) {
      const auto& d = solver.domain(i);
      std::cout << "  " << insem::to_string(d.medium) << " domain:  " << d.dofs.n_elements() << " elements, "
                << d.n_global() << " points\n";
    }
    std::cout << "  dt:              " << solver.dt() << " s (stability limit " << solver.stable_dt() << " s)\n"
              << "  steps:           " << solver.n_steps() << "\n";

    const auto start = std::chrono::steady_clock::now();
    solver.run();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    sim.write_results(results_path);
    std::cout << "  wall time:       " << seconds << " s\n"
              << "  results:         " << results_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "insem-run: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
