#pragma once

#include <array>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "insem/basis/lagrange.hpp"
#include "insem/basis/quadrature.hpp"
#include "insem/core/config.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/mesh/forest.hpp"
#include "insem/ops/dof_map.hpp"
#include "insem/ops/element_block.hpp"
#include "insem/physics/kernels.hpp"
#include "insem/sim/description.hpp"

namespace insem::solver {

// The leaves of one medium, with their own numbering and fields. Elastic
// fields have 2 components (u_x, u_z); acoustic fields have 1 (pressure).
// Global arrays are [component][global point].
struct Domain {
  Medium medium = Medium::Elastic;
  int ncomp = 2;
  std::vector<index_t> leaves;           // forest leaf of each domain element
  std::vector<index_t> element_of_leaf;  // inverse map, -1 for leaves of other domains
  ops::DofMap dofs;
  std::vector<ops::ElementBlock> blocks;

  std::vector<real_t> mass;      // assembled diagonal mass [n_global]
  std::vector<real_t> inv_mass;  // its inverse

  // Stacey: global points and their accumulated coefficients
  // (elastic: cxx, czz, cxz per point; acoustic: c per point).
  std::vector<index_t> absorbing_points;
  std::vector<real_t> absorbing_coef;
  // Global points with a homogeneous Dirichlet condition (rigid elastic,
  // pressure-release acoustic).
  std::vector<index_t> dirichlet_points;

  std::vector<real_t> u, v, a, f;  // field, its first and second time derivative, force
  std::vector<real_t> u_local, f_local;
  std::vector<double> coords;  // [2][n_global] point coordinates

  index_t n_global() const { return dofs.n_global(); }
};

// Inputs of the solver (owned by the Simulation).
struct Problem {
  const mesh::Forest& forest;
  const geometry::Geometry& geometry;
  const basis::Rule& gll;
  const std::map<int, Material>& materials;
  const std::map<int, BoundaryType>& boundaries;
  const std::vector<Source>& sources;
  const std::vector<Receiver>& receivers;
  const TimeConfig& time;
};

// Explicit central-difference time stepping of the semi-discrete system
// M a = f_int(u) + f_src(t) - C v, per domain (docs/DESIGN.md §3.2), with
// fluid-solid coupling: the solid feels the traction p n_f, the fluid the
// normal acceleration -n_f . a_s, so each step solves the solid first.
//   u^{n+1}   = u^n + dt v^n + dt^2/2 a^n
//   v^{n+1/2} = v^n + dt/2 a^n
//   a^{n+1}   = M^-1 [f_int(u^{n+1}) + f_src(t^{n+1}) - C v^{n+1}]
//   v^{n+1}   = v^{n+1/2} + dt/2 a^{n+1}
// The Stacey damping C is local to each boundary point, so the implicit
// velocity in the third line costs a 2x2 solve per point.
class Solver {
 public:
  explicit Solver(const Problem& problem);

  // Time step in use, and the estimated stability limit 2 / sqrt(lambda_max(M^-1 K)).
  double dt() const { return dt_; }
  double stable_dt() const { return stable_dt_; }
  int n_steps() const { return n_steps_; }
  int step() const { return step_; }
  double time() const { return step_ * dt_; }

  // Advances `n` steps (or up to n_steps()); run() advances to the end.
  void advance(int n);
  void run() { advance(n_steps_ - step_); }

  std::size_t n_domains() const { return domains_.size(); }
  const Domain& domain(std::size_t i) const { return domains_[i]; }
  // Index of the domain of a medium, or -1.
  int domain_index(Medium m) const;

  // Initial conditions (field and its time derivative, [comp][n_global]) of
  // the domain of `medium`. Only allowed before the first step. Dirichlet
  // points are forced to zero.
  void set_state(Medium medium, const std::vector<double>& u, const std::vector<double>& v);

  // Receiver traces: [n_samples][ncomp] for quantity q of receiver r.
  struct Trace {
    std::string receiver;
    std::string quantity;
    int ncomp = 0;
    std::vector<double> data;
  };
  const std::vector<Trace>& traces() const { return traces_; }
  const std::vector<double>& sample_times() const { return sample_times_; }

  // Energy monitor samples: time, kinetic and potential energy summed over
  // domains. "Kinetic" is the discrete form 1/2 v.Mv - dt^2/8 a.Ma that the
  // central-difference scheme conserves exactly (without sources and
  // absorbing boundaries). For acoustic domains these are the corresponding
  // quadratic forms of the pressure equation.
  const std::vector<std::array<double, 3>>& energy() const { return energy_; }

  void write_results(const std::string& path) const;

 private:
  struct SourceTerm {
    int domain;
    std::vector<index_t> points;
    std::vector<real_t> coef;  // [comp][point]
    SourceTimeFunction stf;
  };
  struct ReceiverTerm {
    std::string name;
    std::array<double, 2> position{0.0, 0.0};
    int domain;
    std::vector<index_t> points;
    std::vector<real_t> value, d_dx, d_dz;
    double rho;
    // particle velocity integration state (acoustic)
    double prev_ax = 0, prev_az = 0, vel_x = 0, vel_z = 0;
    std::vector<std::size_t> trace_index;  // into traces_, one per quantity
    std::vector<std::string> quantities;
  };

  void build_domains(const Problem& p);
  void build_boundaries(const Problem& p, Domain& d);
  void build_sources(const Problem& p);
  void build_receivers(const Problem& p);
  double estimate_stable_dt();
  double coupled_lambda_max(std::mt19937& rng);

  // f_int(u) into d.f; returns u . f_int when `energy` is set.
  double internal_forces(Domain& d, const std::vector<real_t>& u, bool energy);
  void build_coupling(const Problem& p);
  void compute_acceleration(double t, bool energy, bool damping_implicit);
  void add_coupling_forces(Domain& d);
  // a = M^-1 f with the Stacey and Dirichlet conditions.
  void solve_acceleration(Domain& d, bool damping_implicit);
  double kinetic_energy() const;
  void initialize();
  void record();

  physics::KernelBasis kernel_basis_;
  basis::LagrangeBasis lagrange_;
  std::vector<Domain> domains_;
  std::vector<SourceTerm> sources_;
  std::vector<ReceiverTerm> receivers_;
  // Fluid-solid interface points: fluid and solid global indices and the
  // quadrature weight times the outward fluid normal (summed over faces).
  struct Coupling {
    std::vector<index_t> fluid_points, solid_points;
    std::vector<real_t> cx, cz;
  } coupling_;
  std::vector<Trace> traces_;
  std::vector<double> sample_times_;
  std::vector<std::array<double, 3>> energy_;
  double potential_ = 0.0;

  double dt_ = 0.0;
  double stable_dt_ = 0.0;
  int n_steps_ = 0;
  int step_ = 0;
  int record_every_ = 1;
  int energy_every_ = 0;
  bool initialized_ = false;
};

}  // namespace insem::solver
