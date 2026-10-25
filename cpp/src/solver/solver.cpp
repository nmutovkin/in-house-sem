#include "insem/solver/solver.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "insem/core/error.hpp"
#include "insem/io/hdf5.hpp"
#include "insem/physics/pointwise.hpp"
#include "insem/solver/locate.hpp"

namespace insem::solver {
namespace {

// Natural boundary condition of a medium (what an untagged face does).
BoundaryType natural_condition(Medium m) {
  return m == Medium::Elastic ? BoundaryType::Free : BoundaryType::Rigid;
}

const Material& material_of_leaf(const Problem& p, index_t leaf) {
  const int region = p.forest.macro().material(p.forest.leaf(leaf).tree);
  return p.materials.at(region);
}

// Local (i, j) of point q (0..n) along face f, in counter-clockwise order.
void face_point(int f, int q, int n, int& i, int& j) {
  switch (f) {
    case 0: i = q; j = 0; break;
    case 1: i = n; j = q; break;
    case 2: i = n - q; j = n; break;
    default: i = 0; j = n - q; break;
  }
}

// Quadrature weight times surface Jacobian, and outward unit normal, of
// point q of face f of a leaf (pt = the point's index in the geometry arrays).
void face_measure(const geometry::Geometry& geom, std::size_t pt, int f, double weight, double& w, double& nx,
                  double& nz) {
  // Tangent in counter-clockwise direction from the stored inverse metric:
  // x_xi = eta_z J, z_xi = -eta_x J, x_eta = -xi_z J, z_eta = xi_x J.
  const double jac = geom.jac[pt];
  double tx, tz;
  if (f == 0 || f == 2) {
    tx = geom.eta_z[pt] * jac;
    tz = -geom.eta_x[pt] * jac;
  } else {
    tx = -geom.xi_z[pt] * jac;
    tz = geom.xi_x[pt] * jac;
  }
  if (f >= 2) {
    tx = -tx;
    tz = -tz;
  }
  const double ds = std::hypot(tx, tz);
  w = weight * ds;
  nx = tz / ds;
  nz = -tx / ds;
}

template <class T>
double dot(const std::vector<T>& a, const std::vector<T>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return s;
}

}  // namespace

Solver::Solver(const Problem& p)
    : kernel_basis_(physics::KernelBasis::from_rule(p.gll.points, p.gll.weights)),
      lagrange_(p.gll.points) {
  INSEM_CHECK(p.forest.macro().coord_system() == mesh::CoordSystem::Cartesian,
              "the axisymmetric solver is milestone M3; only Cartesian meshes can be run");
  INSEM_CHECK(p.gll.size() >= physics::kMinNgll && p.gll.size() <= physics::kMaxNgll,
              "polynomial order " << p.gll.size() - 1 << " is not supported by the kernels (1.."
                                  << physics::kMaxNgll - 1 << ")");
  INSEM_CHECK(p.time.record_every >= 1, "record_every must be >= 1");
  INSEM_CHECK(p.time.energy_every >= 0, "energy_every must be >= 0");
  record_every_ = p.time.record_every;
  energy_every_ = p.time.energy_every;

  build_domains(p);
  for (Domain& d : domains_) build_boundaries(p, d);
  build_coupling(p);
  build_sources(p);
  build_receivers(p);

  stable_dt_ = estimate_stable_dt();
  if (p.time.dt > 0.0) {
    INSEM_CHECK(p.time.dt <= stable_dt_, "time step " << p.time.dt << " s exceeds the estimated stability limit "
                                                      << stable_dt_ << " s");
    dt_ = p.time.dt;
  } else {
    INSEM_CHECK(p.time.safety > 0.0 && p.time.safety <= 1.0, "time step safety factor must be in (0, 1]");
    dt_ = p.time.safety * stable_dt_;
  }
  if (p.time.n_steps > 0) {
    n_steps_ = p.time.n_steps;
  } else if (p.time.duration > 0.0) {
    n_steps_ = static_cast<int>(std::ceil(p.time.duration / dt_ - 1e-9));
  }
}

void Solver::build_domains(const Problem& p) {
  const mesh::MacroMesh& macro = p.forest.macro();
  for (index_t c = 0; c < macro.n_cells(); ++c) {
    const auto it = p.materials.find(macro.material(c));
    INSEM_CHECK(it != p.materials.end(), "no material assigned to region " << macro.material(c));
    it->second.validate();
  }
  const int ngll = p.gll.size();
  const std::size_t npe = static_cast<std::size_t>(ngll) * ngll;
  for (Medium medium : {Medium::Acoustic, Medium::Elastic}) {
    Domain d;
    d.medium = medium;
    d.ncomp = medium == Medium::Elastic ? 2 : 1;
    d.element_of_leaf.assign(p.forest.n_leaves(), -1);
    for (index_t l = 0; l < p.forest.n_leaves(); ++l) {
      if (material_of_leaf(p, l).medium == medium) {
        d.element_of_leaf[l] = static_cast<index_t>(d.leaves.size());
        d.leaves.push_back(l);
      }
    }
    if (d.leaves.empty()) continue;
    d.dofs = ops::DofMap::build(p.forest, ngll, d.leaves);

    // Per-point material coefficients and mass density.
    const std::size_t nl = d.leaves.size() * npe;
    std::vector<double> coef0(nl), coef1(nl), local_mass(nl);
    for (std::size_t e = 0; e < d.leaves.size(); ++e) {
      const Material& m = material_of_leaf(p, d.leaves[e]);
      for (std::size_t k = 0; k < npe; ++k) {
        const std::size_t q = e * npe + k;
        const double wj = p.gll.weights[k % ngll] * p.gll.weights[k / ngll] *
                          p.geometry.jac[static_cast<std::size_t>(d.leaves[e]) * npe + k];
        if (medium == Medium::Elastic) {
          coef0[q] = m.rho * (m.vp * m.vp - 2.0 * m.vs * m.vs);
          coef1[q] = m.rho * m.vs * m.vs;
          local_mass[q] = wj * m.rho;
        } else {
          coef0[q] = 1.0 / m.rho;
          coef1[q] = 0.0;
          local_mass[q] = wj / (m.rho * m.vp * m.vp);
        }
      }
    }
    d.blocks = ops::build_blocks(d.leaves, p.geometry, coef0, coef1);
    std::vector<double> mass(d.n_global());
    d.dofs.assemble(local_mass.data(), mass.data(), 1);
    d.mass.assign(mass.begin(), mass.end());
    d.inv_mass.resize(mass.size());
    for (std::size_t g = 0; g < mass.size(); ++g) d.inv_mass[g] = static_cast<real_t>(1.0 / mass[g]);

    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    for (auto* v : {&d.u, &d.v, &d.a, &d.f}) v->assign(d.ncomp * ng, real_t(0));
    d.u_local.assign(d.ncomp * nl, real_t(0));
    d.f_local.assign(d.ncomp * nl, real_t(0));

    d.coords.resize(2 * ng);
    const auto& offsets = d.dofs.assembly_offsets();
    const auto& slots = d.dofs.assembly_slots();
    for (std::size_t g = 0; g < ng; ++g) {
      const std::size_t s = static_cast<std::size_t>(slots[offsets[g]]);
      const std::size_t q = static_cast<std::size_t>(d.leaves[s / npe]) * npe + s % npe;
      d.coords[g] = p.geometry.x[q];
      d.coords[ng + g] = p.geometry.z[q];
    }
    domains_.push_back(std::move(d));
  }
}

void Solver::build_boundaries(const Problem& p, Domain& d) {
  const mesh::MacroMesh& macro = p.forest.macro();
  const geometry::Geometry& geom = p.geometry;
  std::map<std::int64_t, int> face_tag;  // cell * 4 + face -> tag
  for (const auto& bf : macro.boundary()) face_tag[static_cast<std::int64_t>(bf.cell) * 4 + bf.face] = bf.tag;

  const int ngll = geom.ngll;
  const int n = ngll - 1;
  const std::size_t npe = static_cast<std::size_t>(ngll) * ngll;
  std::map<index_t, std::array<double, 3>> absorbing;  // ordered: deterministic
  std::set<index_t> dirichlet;

  for (std::size_t e = 0; e < d.leaves.size(); ++e) {
    const index_t leaf = d.leaves[e];
    const mesh::Leaf& lf = p.forest.leaf(leaf);
    const int last = (1 << lf.level) - 1;
    const bool on_macro_face[4] = {lf.j == 0, lf.i == last, lf.j == last, lf.i == 0};
    const Material& mat = material_of_leaf(p, leaf);
    for (int f = 0; f < 4; ++f) {
      if (!on_macro_face[f] || !macro.is_boundary_edge(macro.cell_edge(lf.tree, f))) continue;
      BoundaryType type = natural_condition(d.medium);
      const auto tag = face_tag.find(static_cast<std::int64_t>(lf.tree) * 4 + f);
      if (tag != face_tag.end()) {
        const auto bc = p.boundaries.find(tag->second);
        if (bc != p.boundaries.end()) type = bc->second;
      }
      const bool is_dirichlet = (d.medium == Medium::Elastic && type == BoundaryType::Rigid) ||
                                (d.medium == Medium::Acoustic && type == BoundaryType::Free);
      if (type != BoundaryType::Absorbing && !is_dirichlet) continue;
      for (int q = 0; q <= n; ++q) {
        int i, j;
        face_point(f, q, n, i, j);
        const int k = j * ngll + i;
        const index_t g = d.dofs.global_index(static_cast<index_t>(e), k);
        if (is_dirichlet) {
          dirichlet.insert(g);
          continue;
        }
        double w, nx, nz;
        face_measure(geom, static_cast<std::size_t>(leaf) * npe + k, f, p.gll.weights[q], w, nx, nz);
        auto& acc = absorbing[g];
        if (d.medium == Medium::Elastic) {
          double cxx, czz, cxz;
          physics::stacey_elastic(w, mat.rho, mat.vp, mat.vs, nx, nz, cxx, czz, cxz);
          acc[0] += cxx;
          acc[1] += czz;
          acc[2] += cxz;
        } else {
          acc[0] += physics::stacey_acoustic(w, mat.rho, mat.vp);
        }
      }
    }
  }
  const int stride = d.medium == Medium::Elastic ? 3 : 1;
  for (const auto& [g, c] : absorbing) {
    if (dirichlet.count(g)) continue;  // Dirichlet wins at shared corners
    d.absorbing_points.push_back(g);
    for (int s = 0; s < stride; ++s) d.absorbing_coef.push_back(static_cast<real_t>(c[s]));
  }
  d.dirichlet_points.assign(dirichlet.begin(), dirichlet.end());
}

void Solver::build_coupling(const Problem& p) {
  const int fi = domain_index(Medium::Acoustic);
  const int si = domain_index(Medium::Elastic);
  if (fi < 0 || si < 0) return;
  const Domain& fluid = domains_[fi];
  const Domain& solid = domains_[si];
  const int ngll = p.gll.size();
  const int n = ngll - 1;
  const std::size_t npe = static_cast<std::size_t>(ngll) * ngll;
  const mesh::LeafVertices lv = p.forest.leaf_vertices();
  auto edge_key = [](index_t a, index_t b) {
    return (static_cast<std::uint64_t>(std::min(a, b)) << 32) | static_cast<std::uint32_t>(std::max(a, b));
  };
  // Faces of solid elements by their leaf edge.
  std::map<std::uint64_t, std::pair<index_t, int>> solid_faces;
  for (std::size_t e = 0; e < solid.leaves.size(); ++e) {
    const index_t* v = &lv.corners[4 * static_cast<std::size_t>(solid.leaves[e])];
    for (int f = 0; f < 4; ++f) {
      solid_faces[edge_key(v[mesh::kFaceCorners[f][0]], v[mesh::kFaceCorners[f][1]])] = {static_cast<index_t>(e), f};
    }
  }
  // Interface points, keyed by fluid global point; faces meeting at a point
  // add up their weighted normals.
  struct Entry {
    index_t solid_point;
    double cx, cz;
  };
  std::map<index_t, Entry> entries;
  for (std::size_t e = 0; e < fluid.leaves.size(); ++e) {
    const index_t leaf = fluid.leaves[e];
    const index_t* v = &lv.corners[4 * static_cast<std::size_t>(leaf)];
    for (int f = 0; f < 4; ++f) {
      const index_t a = v[mesh::kFaceCorners[f][0]], b = v[mesh::kFaceCorners[f][1]];
      const auto it = solid_faces.find(edge_key(a, b));
      if (it == solid_faces.end()) continue;
      const auto [es, fs] = it->second;
      const index_t* vs = &lv.corners[4 * static_cast<std::size_t>(solid.leaves[es])];
      const bool solid_forward = vs[mesh::kFaceCorners[fs][0]] < vs[mesh::kFaceCorners[fs][1]];
      for (int q = 0; q <= n; ++q) {
        int i, j;
        face_point(f, q, n, i, j);
        const int k = j * ngll + i;
        const int pos = a < b ? q : n - q;  // canonical position along the edge
        int is, js;
        face_point(fs, solid_forward ? pos : n - pos, n, is, js);
        const index_t gf = fluid.dofs.global_index(static_cast<index_t>(e), k);
        const index_t gs = solid.dofs.global_index(es, js * ngll + is);
        double w, nx, nz;
        face_measure(p.geometry, static_cast<std::size_t>(leaf) * npe + k, f, p.gll.weights[q], w, nx, nz);
        auto [entry, inserted] = entries.try_emplace(gf, Entry{gs, 0.0, 0.0});
        INSEM_CHECK(entry->second.solid_point == gs, "inconsistent fluid-solid interface point");
        entry->second.cx += w * nx;
        entry->second.cz += w * nz;
      }
    }
  }
  for (const auto& [gf, en] : entries) {
    coupling_.fluid_points.push_back(gf);
    coupling_.solid_points.push_back(en.solid_point);
    coupling_.cx.push_back(static_cast<real_t>(en.cx));
    coupling_.cz.push_back(static_cast<real_t>(en.cz));
  }
}

int Solver::domain_index(Medium m) const {
  for (std::size_t i = 0; i < domains_.size(); ++i) {
    if (domains_[i].medium == m) return static_cast<int>(i);
  }
  return -1;
}

void Solver::build_sources(const Problem& p) {
  for (std::size_t si = 0; si < p.sources.size(); ++si) {
    const Source& src = p.sources[si];
    const int di = domain_index(src.medium());
    INSEM_CHECK(di >= 0, "source " << si << " (" << src.kind_name() << ") needs an " << to_string(src.medium())
                                   << " region, but the model has none");
    const Domain& d = domains_[di];
    const auto loc = locate(p.forest, p.geometry, src.position, d.leaves);
    INSEM_CHECK(loc, "source " << si << " at (" << src.position[0] << ", " << src.position[1]
                               << ") is not inside the " << to_string(src.medium()) << " part of the model");
    const PointWeights w = point_weights(p.forest, lagrange_, *loc);
    const index_t e = d.element_of_leaf[loc->leaf];
    const int npe = d.dofs.points_per_element();

    SourceTerm term;
    term.domain = di;
    term.stf = src.stf;
    term.coef.assign(static_cast<std::size_t>(d.ncomp) * npe, real_t(0));
    for (int k = 0; k < npe; ++k) term.points.push_back(d.dofs.global_index(e, k));
    const auto& c = src.components;
    for (int k = 0; k < npe; ++k) {
      switch (src.kind) {
        case Source::Kind::Force:
          term.coef[k] = static_cast<real_t>(c[0] * w.value[k]);
          term.coef[npe + k] = static_cast<real_t>(c[1] * w.value[k]);
          break;
        case Source::Kind::MomentTensor:
          // Equivalent body force -div(M delta): f_a(p) = M_ab d_b phi_p(x_s).
          term.coef[k] = static_cast<real_t>(c[0] * w.d_dx[k] + c[2] * w.d_dz[k]);
          term.coef[npe + k] = static_cast<real_t>(c[2] * w.d_dx[k] + c[1] * w.d_dz[k]);
          break;
        case Source::Kind::Monopole:
          term.coef[k] = static_cast<real_t>(w.value[k]);
          break;
      }
    }
    sources_.push_back(std::move(term));
  }
}

void Solver::build_receivers(const Problem& p) {
  for (const Receiver& rec : p.receivers) {
    ReceiverTerm term;
    term.name = rec.name;
    term.position = rec.position;
    std::optional<PointLocation> loc;
    for (std::size_t di = 0; di < domains_.size() && !loc; ++di) {
      if (!rec.medium.empty() && to_string(domains_[di].medium) != rec.medium) continue;
      loc = locate(p.forest, p.geometry, rec.position, domains_[di].leaves);
      if (loc) term.domain = static_cast<int>(di);
    }
    INSEM_CHECK(loc, "receiver '" << rec.name << "' at (" << rec.position[0] << ", " << rec.position[1]
                                  << ") is outside the " << (rec.medium.empty() ? "" : rec.medium + " part of the ")
                                  << "model");
    const Domain& d = domains_[term.domain];
    const PointWeights w = point_weights(p.forest, lagrange_, *loc);
    const index_t e = d.element_of_leaf[loc->leaf];
    for (int k = 0; k < d.dofs.points_per_element(); ++k) {
      term.points.push_back(d.dofs.global_index(e, k));
      term.value.push_back(static_cast<real_t>(w.value[k]));
      term.d_dx.push_back(static_cast<real_t>(w.d_dx[k]));
      term.d_dz.push_back(static_cast<real_t>(w.d_dz[k]));
    }
    term.rho = material_of_leaf(p, loc->leaf).rho;
    INSEM_CHECK(!rec.quantities.empty(), "receiver '" << rec.name << "' records no quantities");
    for (const std::string& q : rec.quantities) {
      Trace t;
      t.receiver = rec.name;
      t.quantity = q;
      t.ncomp = quantity_components(q, d.medium);
      term.trace_index.push_back(traces_.size());
      term.quantities.push_back(q);
      traces_.push_back(std::move(t));
    }
    receivers_.push_back(std::move(term));
  }
}

double Solver::internal_forces(Domain& d, const std::vector<real_t>& u, bool energy) {
  d.dofs.gather(u.data(), d.u_local.data(), d.ncomp);
  for (const auto& blk : d.blocks) {
    if (d.medium == Medium::Elastic) {
      physics::elastic_internal_forces(kernel_basis_, blk, d.u_local.data(), d.f_local.data());
    } else {
      physics::acoustic_internal_forces(kernel_basis_, blk, d.u_local.data(), d.f_local.data());
    }
  }
  d.dofs.assemble(d.f_local.data(), d.f.data(), d.ncomp);
  return energy ? dot(u, d.f) : 0.0;
}

double Solver::estimate_stable_dt() {
  // Power iteration for lambda_max(M^-1 K), with the Rayleigh quotient
  // x.Kx / x.Mx as the estimate (largest value seen).
  double lambda = 0.0;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (Domain& d : domains_) {
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    std::vector<real_t> x(d.ncomp * ng);
    for (auto& v : x) v = static_cast<real_t>(dist(rng));
    for (int it = 0; it < 100; ++it) {
      for (int c = 0; c < d.ncomp; ++c) {
        for (index_t g : d.dirichlet_points) x[c * ng + g] = 0;
      }
      const double xkx = -internal_forces(d, x, true);
      double xmx = 0.0, norm = 0.0;
      for (int c = 0; c < d.ncomp; ++c) {
        for (std::size_t g = 0; g < ng; ++g) {
          xmx += static_cast<double>(d.mass[g]) * x[c * ng + g] * x[c * ng + g];
        }
      }
      if (xmx > 0.0) lambda = std::max(lambda, xkx / xmx);
      for (int c = 0; c < d.ncomp; ++c) {
        for (std::size_t g = 0; g < ng; ++g) {
          const double y = -static_cast<double>(d.f[c * ng + g]) * d.inv_mass[g];
          x[c * ng + g] = static_cast<real_t>(y);
          norm += y * y;
        }
      }
      if (norm == 0.0) break;
      const double scale = 1.0 / std::sqrt(norm);
      for (auto& v : x) v = static_cast<real_t>(v * scale);
    }
    std::fill(d.f.begin(), d.f.end(), real_t(0));
  }
  if (!coupling_.fluid_points.empty()) lambda = std::max(lambda, coupled_lambda_max(rng));
  INSEM_CHECK(lambda > 0.0, "could not estimate the stable time step (empty or rigid model?)");
  return 2.0 / std::sqrt(lambda);
}

double Solver::coupled_lambda_max(std::mt19937& rng) {
  // The coupled operator x -> -a(x) is not symmetric, so its spectral radius
  // is estimated from the growth of the mass-weighted norm under power
  // iteration (no sources, no damping). The domain fields are restored.
  Domain& solid = domains_[domain_index(Medium::Elastic)];
  Domain& fluid = domains_[domain_index(Medium::Acoustic)];
  const auto saved_su = solid.u, saved_fu = fluid.u, saved_sa = solid.a, saved_fa = fluid.a;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& v : solid.u) v = static_cast<real_t>(dist(rng));
  for (auto& v : fluid.u) v = static_cast<real_t>(dist(rng));
  auto mass_norm2 = [](const Domain& d, const std::vector<real_t>& x) {
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    double s = 0.0;
    for (int c = 0; c < d.ncomp; ++c) {
      for (std::size_t g = 0; g < ng; ++g) s += static_cast<double>(d.mass[g]) * x[c * ng + g] * x[c * ng + g];
    }
    return s;
  };
  double lambda = 0.0;
  for (int it = 0; it < 120; ++it) {
    for (Domain* d : {&solid, &fluid}) {
      const std::size_t ng = static_cast<std::size_t>(d->n_global());
      for (int c = 0; c < d->ncomp; ++c) {
        for (index_t g : d->dirichlet_points) d->u[c * ng + g] = 0;
      }
    }
    const double nx = mass_norm2(solid, solid.u) + mass_norm2(fluid, fluid.u);
    for (Domain* d : {&solid, &fluid}) {
      internal_forces(*d, d->u, false);
      add_coupling_forces(*d);
      const std::size_t ng = static_cast<std::size_t>(d->n_global());
      for (int c = 0; c < d->ncomp; ++c) {
        for (std::size_t g = 0; g < ng; ++g) d->a[c * ng + g] = d->f[c * ng + g] * d->inv_mass[g];
        for (index_t g : d->dirichlet_points) d->a[c * ng + g] = 0;
      }
    }
    const double ny = mass_norm2(solid, solid.a) + mass_norm2(fluid, fluid.a);
    if (nx <= 0.0 || ny <= 0.0) break;
    if (it >= 60) lambda = std::max(lambda, std::sqrt(ny / nx));
    const double scale = -1.0 / std::sqrt(ny);
    for (Domain* d : {&solid, &fluid}) {
      for (std::size_t i = 0; i < d->u.size(); ++i) d->u[i] = static_cast<real_t>(d->a[i] * scale);
    }
  }
  solid.u = saved_su;
  fluid.u = saved_fu;
  solid.a = saved_sa;
  fluid.a = saved_fa;
  std::fill(solid.f.begin(), solid.f.end(), real_t(0));
  std::fill(fluid.f.begin(), fluid.f.end(), real_t(0));
  return lambda;
}

void Solver::compute_acceleration(double t, bool energy, bool damping_implicit) {
  // The solid is solved first: its coupling traction needs the current
  // pressure, while the fluid's coupling term needs the new solid
  // acceleration (docs/DESIGN.md §2.3).
  potential_ = 0.0;
  for (Medium medium : {Medium::Elastic, Medium::Acoustic}) {
    const int di = domain_index(medium);
    if (di < 0) continue;
    Domain& d = domains_[di];
    potential_ -= 0.5 * internal_forces(d, d.u, energy);
    for (const SourceTerm& s : sources_) {
      if (s.domain != di) continue;
      const real_t amp = static_cast<real_t>(s.stf(t));
      const std::size_t ng = static_cast<std::size_t>(d.n_global());
      const std::size_t np = s.points.size();
      for (int c = 0; c < d.ncomp; ++c) {
        for (std::size_t i = 0; i < np; ++i) d.f[c * ng + s.points[i]] += s.coef[c * np + i] * amp;
      }
    }
    add_coupling_forces(d);
    solve_acceleration(d, damping_implicit);
  }
}

void Solver::add_coupling_forces(Domain& d) {
  const std::size_t n = coupling_.fluid_points.size();
  if (n == 0) return;
  if (d.medium == Medium::Elastic) {
    // Traction p n_f on the solid.
    const Domain& fluid = domains_[domain_index(Medium::Acoustic)];
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    for (std::size_t i = 0; i < n; ++i) {
      const real_t p = fluid.u[coupling_.fluid_points[i]];
      const std::size_t g = coupling_.solid_points[i];
      d.f[g] += coupling_.cx[i] * p;
      d.f[ng + g] += coupling_.cz[i] * p;
    }
  } else {
    // -n_f . a_s on the fluid.
    const Domain& solid = domains_[domain_index(Medium::Elastic)];
    const std::size_t ngs = static_cast<std::size_t>(solid.n_global());
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t gs = coupling_.solid_points[i];
      d.f[coupling_.fluid_points[i]] -= coupling_.cx[i] * solid.a[gs] + coupling_.cz[i] * solid.a[ngs + gs];
    }
  }
}

void Solver::solve_acceleration(Domain& d, bool damping_implicit) {
  const std::size_t ng = static_cast<std::size_t>(d.n_global());
  const int ncomp = d.ncomp;
  const real_t* inv_mass = d.inv_mass.data();
  const real_t* f = d.f.data();
  real_t* a = d.a.data();
#pragma omp parallel for schedule(static)
  for (std::int64_t g = 0; g < static_cast<std::int64_t>(ng); ++g) {
    for (int c = 0; c < ncomp; ++c) a[c * ng + g] = f[c * ng + g] * inv_mass[g];
  }

  // Stacey points: the damping force -C v uses v^{n+1} = v^{n+1/2} + dt/2 a,
  // i.e. (M + dt/2 C) a = f - C v^{n+1/2}, solved per point (C is 2x2 or
  // scalar). Treating the damping implicitly keeps it unconditionally
  // stable; at t = 0 the velocity is known and the term is explicit.
  const double h = damping_implicit ? 0.5 * dt_ : 0.0;
  const std::size_t nabs = d.absorbing_points.size();
  if (d.medium == Medium::Elastic) {
    for (std::size_t i = 0; i < nabs; ++i) {
      const std::size_t g = d.absorbing_points[i];
      const double cxx = d.absorbing_coef[3 * i], czz = d.absorbing_coef[3 * i + 1],
                   cxz = d.absorbing_coef[3 * i + 2];
      const double vx = d.v[g], vz = d.v[ng + g];
      const double rx = d.f[g] - (cxx * vx + cxz * vz);
      const double rz = d.f[ng + g] - (cxz * vx + czz * vz);
      const double m = d.mass[g];
      const double a11 = m + h * cxx, a22 = m + h * czz, a12 = h * cxz;
      const double det = a11 * a22 - a12 * a12;
      d.a[g] = static_cast<real_t>((a22 * rx - a12 * rz) / det);
      d.a[ng + g] = static_cast<real_t>((a11 * rz - a12 * rx) / det);
    }
  } else {
    for (std::size_t i = 0; i < nabs; ++i) {
      const std::size_t g = d.absorbing_points[i];
      const double c = d.absorbing_coef[i];
      d.a[g] = static_cast<real_t>((d.f[g] - c * d.v[g]) / (d.mass[g] + h * c));
    }
  }
  for (int c = 0; c < ncomp; ++c) {
    for (index_t g : d.dirichlet_points) d.a[c * ng + g] = 0;
  }
}

void Solver::set_state(Medium medium, const std::vector<double>& u, const std::vector<double>& v) {
  INSEM_CHECK(!initialized_, "initial conditions must be set before the first time step");
  const int di = domain_index(medium);
  INSEM_CHECK(di >= 0, "the model has no " << to_string(medium) << " region");
  Domain& d = domains_[di];
  const std::size_t n = static_cast<std::size_t>(d.ncomp) * d.n_global();
  INSEM_CHECK(u.size() == n && v.size() == n,
              "state arrays must have " << d.ncomp << " x " << d.n_global() << " entries");
  for (std::size_t i = 0; i < n; ++i) {
    d.u[i] = static_cast<real_t>(u[i]);
    d.v[i] = static_cast<real_t>(v[i]);
  }
  const std::size_t ng = static_cast<std::size_t>(d.n_global());
  for (int c = 0; c < d.ncomp; ++c) {
    for (index_t g : d.dirichlet_points) d.u[c * ng + g] = d.v[c * ng + g] = 0;
  }
}

void Solver::record() {
  sample_times_.push_back(time());
  for (ReceiverTerm& r : receivers_) {
    const Domain& d = domains_[r.domain];
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    const std::size_t np = r.points.size();
    auto interpolate = [&](const std::vector<real_t>& field, int c) {
      double s = 0.0;
      for (std::size_t i = 0; i < np; ++i) s += static_cast<double>(r.value[i]) * field[c * ng + r.points[i]];
      return s;
    };
    for (std::size_t qi = 0; qi < r.quantities.size(); ++qi) {
      Trace& t = traces_[r.trace_index[qi]];
      const std::string& q = r.quantities[qi];
      if (q == "particle_velocity") {
        t.data.push_back(r.vel_x);
        t.data.push_back(r.vel_z);
        continue;
      }
      const std::vector<real_t>& field = (q == "velocity") ? d.v : (q == "acceleration") ? d.a : d.u;
      for (int c = 0; c < t.ncomp; ++c) t.data.push_back(interpolate(field, c));
    }
  }
}

void Solver::initialize() {
  const bool energy = energy_every_ > 0;
  compute_acceleration(0.0, energy, false);
  // Fluid particle acceleration -grad(p) / rho at the receivers, for the
  // trapezoidal integration of particle velocity.
  for (ReceiverTerm& r : receivers_) {
    const Domain& d = domains_[r.domain];
    if (d.medium != Medium::Acoustic) continue;
    double gx = 0.0, gz = 0.0;
    for (std::size_t i = 0; i < r.points.size(); ++i) {
      gx += static_cast<double>(r.d_dx[i]) * d.u[r.points[i]];
      gz += static_cast<double>(r.d_dz[i]) * d.u[r.points[i]];
    }
    r.prev_ax = -gx / r.rho;
    r.prev_az = -gz / r.rho;
  }
  initialized_ = true;
  record();
  if (energy) energy_.push_back({0.0, kinetic_energy(), potential_});
}

double Solver::kinetic_energy() const {
  // Kinetic part of the energy that central differences conserve exactly:
  // 1/2 v+ . M v- with v+- = v -+ dt/2 a, i.e. 1/2 v.Mv - dt^2/8 a.Ma.
  const double c = dt_ * dt_ / 8.0;
  double kinetic = 0.0;
  for (const Domain& d : domains_) {
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    for (int comp = 0; comp < d.ncomp; ++comp) {
      for (std::size_t g = 0; g < ng; ++g) {
        const double v = d.v[comp * ng + g], a = d.a[comp * ng + g];
        kinetic += d.mass[g] * (0.5 * v * v - c * a * a);
      }
    }
  }
  return kinetic;
}

void Solver::advance(int n) {
  if (!initialized_) initialize();
  const double dt = dt_;
  const real_t rdt = static_cast<real_t>(dt);
  const real_t half_dt = static_cast<real_t>(0.5 * dt);
  const real_t half_dt2 = static_cast<real_t>(0.5 * dt * dt);
  for (int s = 0; s < n && step_ < n_steps_; ++s) {
    for (Domain& d : domains_) {
      real_t* u = d.u.data();
      real_t* v = d.v.data();
      const real_t* a = d.a.data();
      const std::int64_t len = static_cast<std::int64_t>(d.u.size());
#pragma omp parallel for schedule(static)
      for (std::int64_t i = 0; i < len; ++i) {
        u[i] += rdt * v[i] + half_dt2 * a[i];
        v[i] += half_dt * a[i];
      }
    }
    ++step_;
    const bool energy = energy_every_ > 0 && step_ % energy_every_ == 0;
    compute_acceleration(step_ * dt, energy, true);
    for (Domain& d : domains_) {
      real_t* v = d.v.data();
      const real_t* a = d.a.data();
      const std::int64_t len = static_cast<std::int64_t>(d.v.size());
#pragma omp parallel for schedule(static)
      for (std::int64_t i = 0; i < len; ++i) v[i] += half_dt * a[i];
    }
    for (ReceiverTerm& r : receivers_) {
      const Domain& d = domains_[r.domain];
      if (d.medium != Medium::Acoustic) continue;
      double gx = 0.0, gz = 0.0;
      for (std::size_t i = 0; i < r.points.size(); ++i) {
        gx += static_cast<double>(r.d_dx[i]) * d.u[r.points[i]];
        gz += static_cast<double>(r.d_dz[i]) * d.u[r.points[i]];
      }
      const double ax = -gx / r.rho, az = -gz / r.rho;
      r.vel_x += 0.5 * dt * (r.prev_ax + ax);
      r.vel_z += 0.5 * dt * (r.prev_az + az);
      r.prev_ax = ax;
      r.prev_az = az;
    }
    if (step_ % record_every_ == 0) record();
    if (energy) energy_.push_back({step_ * dt, kinetic_energy(), potential_});
  }
}

void Solver::write_results(const std::string& path) const {
  io::H5File f(path, io::H5File::Mode::Truncate);
  f.set_attribute("/", "format", std::string("insem-results"));
  f.set_attribute("/", "version", std::int64_t{1});
  f.set_attribute("/", "insem_version", std::string(insem::version()));
  f.set_attribute("/", "dt", dt_);
  f.set_attribute("/", "n_steps", static_cast<std::int64_t>(n_steps_));
  f.set_attribute("/", "steps_done", static_cast<std::int64_t>(step_));
  f.set_attribute("/", "record_every", static_cast<std::int64_t>(record_every_));
  f.write("/time", sample_times_);
  for (const Trace& t : traces_) {
    const std::size_t n = t.data.size() / static_cast<std::size_t>(t.ncomp);
    f.write("/receivers/" + t.receiver + "/" + t.quantity, t.data.data(), {n, static_cast<std::size_t>(t.ncomp)});
  }
  for (const ReceiverTerm& r : receivers_) {
    const std::string group = "/receivers/" + r.name;
    f.set_attribute(group, "medium", to_string(domains_[r.domain].medium));
    f.set_attribute(group, "x", r.position[0]);
    f.set_attribute(group, "z", r.position[1]);
  }
  if (!energy_.empty()) {
    std::vector<double> e;
    for (const auto& row : energy_) e.insert(e.end(), row.begin(), row.end());
    f.write("/energy", e.data(), {energy_.size(), 3});
    f.set_attribute("/energy", "columns", std::string("time,kinetic,potential"));
  }
}

}  // namespace insem::solver
