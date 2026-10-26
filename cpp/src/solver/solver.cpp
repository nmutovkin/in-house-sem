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

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtHalf = 0.70710678118654752440;

// Solves A x = b in place (b becomes x) for a small symmetric positive
// definite A (n <= 3, row-major), by Gaussian elimination without pivoting.
void solve_spd(int n, double* A, double* b) {
  for (int k = 0; k < n; ++k) {
    for (int i = k + 1; i < n; ++i) {
      const double l = A[i * n + k] / A[k * n + k];
      for (int j = k; j < n; ++j) A[i * n + j] -= l * A[k * n + j];
      b[i] -= l * b[k];
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    double s = b[k];
    for (int j = k + 1; j < n; ++j) s -= A[k * n + j] * b[j];
    b[k] = s / A[k * n + k];
  }
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
      lagrange_(p.gll.points),
      glj_(basis::glj01(p.gll.size())),
      lagrange_glj_(glj_.points),
      axisym_(p.forest.macro().coord_system() == mesh::CoordSystem::Axisymmetric),
      azimuthal_order_(p.azimuthal_order) {
  INSEM_CHECK(azimuthal_order_ == 0 || (azimuthal_order_ == 1 && axisym_),
              "azimuthal order " << azimuthal_order_ << " needs an axisymmetric mesh (supported orders: 0, 1)");
  INSEM_CHECK(p.gll.size() >= physics::kMinNgll && p.gll.size() <= physics::kMaxNgll,
              "polynomial order " << p.gll.size() - 1 << " is not supported by the kernels (1.."
                                  << physics::kMaxNgll - 1 << ")");
  INSEM_CHECK(p.time.record_every >= 1, "record_every must be >= 1");
  INSEM_CHECK(p.time.energy_every >= 0, "energy_every must be >= 0");
  record_every_ = p.time.record_every;
  energy_every_ = p.time.energy_every;
  INSEM_CHECK(p.time.snapshot_every >= 0, "snapshot_every must be >= 0");
  snapshot_every_ = p.time.snapshot_every;
  geometry_x_ = p.geometry.x;
  geometry_z_ = p.geometry.z;

  build_domains(p);
  for (Domain& d : domains_) build_boundaries(p, d);
  build_coupling(p);
  build_sources(p);
  build_receivers(p);
  if (snapshot_every_ > 0) {
    for (const std::string& q : p.time.snapshot_quantities) {
      INSEM_CHECK(q == "displacement" || q == "velocity" || q == "acceleration" || q == "pressure",
                  "unknown snapshot quantity '" << q << "' (displacement, velocity, acceleration, pressure)");
      for (const Domain& d : domains_) {
        const bool exists = d.medium == Medium::Elastic
                                ? (q == "displacement" || q == "velocity" || q == "acceleration")
                                : q == "pressure";
        if (!exists) continue;
        snapshots_.push_back({d.medium, q, d.ncomp, {}});
      }
    }
  }

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
    d.azimuthal_order = azimuthal_order_;
    d.ncomp = medium == Medium::Elastic ? (azimuthal_order_ == 1 ? 3 : 2) : 1;
    if (axisym_) {
      if (medium == Medium::Elastic) {
        d.axis_components = azimuthal_order_ == 1 ? std::vector<int>{0, 2} : std::vector<int>{0};
      } else if (azimuthal_order_ == 1) {
        d.axis_components = {0};
      }
    }
    d.element_of_leaf.assign(p.forest.n_leaves(), -1);
    for (index_t l = 0; l < p.forest.n_leaves(); ++l) {
      if (material_of_leaf(p, l).medium == medium) {
        d.element_of_leaf[l] = static_cast<index_t>(d.leaves.size());
        d.leaves.push_back(l);
      }
    }
    if (d.leaves.empty()) continue;
    d.dofs = ops::DofMap::build(p.forest, ngll, d.leaves);

    // Per-point quadrature weights. Axisymmetric: weight x J x r, where
    // axial elements integrate r with the GLJ rule along xi:
    //   int f r dxi = int f [r / (1 + xi)] (1 + xi) dxi ~ sum_i w^J_i f_i rt_i,
    // rt_i = r_i / (1 + xi_i), and on the axis rt = dr/dxi (its limit). The
    // hoop term carries weight x J (= wr / r off the axis).
    const std::size_t nl = d.leaves.size() * npe;
    std::vector<double> wr(nl), hoop(nl, 0.0);
    std::vector<char> axial(d.leaves.size(), 0);
    for (std::size_t e = 0; e < d.leaves.size(); ++e) {
      const index_t leaf = d.leaves[e];
      axial[e] = axisym_ && p.forest.is_axial(leaf) ? 1 : 0;
      for (std::size_t k = 0; k < npe; ++k) {
        const int i = static_cast<int>(k % ngll), j = static_cast<int>(k / ngll);
        const std::size_t pt = static_cast<std::size_t>(leaf) * npe + k;
        const double jac = p.geometry.jac[pt];
        if (!axisym_) {
          wr[e * npe + k] = p.gll.weights[i] * p.gll.weights[j] * jac;
        } else if (!axial[e]) {
          hoop[e * npe + k] = p.gll.weights[i] * p.gll.weights[j] * jac;
          wr[e * npe + k] = hoop[e * npe + k] * std::max(p.geometry.x[pt], 0.0);
        } else {
          const double x_xi = p.geometry.eta_z[pt] * jac;  // dr/dxi
          const double rt = i == 0 ? x_xi : p.geometry.x[pt] / (1.0 + glj_.points[i]);
          wr[e * npe + k] = glj_.weights[i] * p.gll.weights[j] * jac * rt;
          hoop[e * npe + k] = i == 0 ? 0.0 : wr[e * npe + k] / p.geometry.x[pt];
        }
      }
    }

    // Per-point material coefficients and mass density.
    std::vector<double> coef0(nl), coef1(nl), local_mass(nl);
    for (std::size_t e = 0; e < d.leaves.size(); ++e) {
      const Material& m = material_of_leaf(p, d.leaves[e]);
      for (std::size_t k = 0; k < npe; ++k) {
        const std::size_t q = e * npe + k;
        const double wj = wr[q];
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
    if (axisym_) {
      d.blocks = {ops::build_axisym_block(d.leaves, p.geometry, coef0, coef1, wr, hoop, axial, azimuthal_order_)};
    } else {
      d.blocks = ops::build_blocks(d.leaves, p.geometry, coef0, coef1);
    }
    std::vector<double> mass(d.n_global());
    d.dofs.assemble(local_mass.data(), mass.data(), 1);
    d.mass.assign(mass.begin(), mass.end());
    d.inv_mass.resize(mass.size());
    for (std::size_t g = 0; g < mass.size(); ++g) {
      INSEM_CHECK(mass[g] > 0.0, "non-positive lumped mass at a global point (degenerate element?)");
      d.inv_mass[g] = static_cast<real_t>(1.0 / mass[g]);
    }

    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    for (auto* v : {&d.u, &d.v, &d.a, &d.f}) v->assign(d.ncomp * ng, real_t(0));
    d.u_local.assign(d.ncomp * nl, real_t(0));
    d.f_local.assign(d.ncomp * nl, real_t(0));

    d.coords.resize(2 * ng);
    for (std::size_t g = 0; g < ng; ++g) {
      const std::size_t s = static_cast<std::size_t>(d.dofs.primary_slot(static_cast<index_t>(g)));
      const std::size_t q = static_cast<std::size_t>(d.leaves[s / npe]) * npe + s % npe;
      d.coords[g] = p.geometry.x[q];
      d.coords[ng + g] = p.geometry.z[q];
      if (!d.axis_components.empty() && p.geometry.x[q] <= 0.0) d.axis_points.push_back(static_cast<index_t>(g));
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
  const int nc = d.ncomp;
  std::map<index_t, std::array<double, 9>> absorbing;  // ordered: deterministic; [nc][nc]
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
        INSEM_CHECK(!(macro.is_axis_face(lf.tree, f) && bc != p.boundaries.end()),
                    "boundary tag " << tag->second << " is on the symmetry axis, which takes no boundary "
                                    "condition (remove it from the boundary settings)");
      }
      if (macro.is_axis_face(lf.tree, f)) continue;  // symmetry axis
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
        face_weight(p, leaf, f, i, j, w, nx, nz);
        auto& acc = absorbing[g];
        if (d.medium == Medium::Acoustic) {
          acc[0] += physics::stacey_acoustic(w, mat.rho, mat.vp);
          continue;
        }
        double cxx, czz, cxz;
        physics::stacey_elastic(w, mat.rho, mat.vp, mat.vs, nx, nz, cxx, czz, cxz);
        if (nc == 2) {
          acc[0] += cxx;
          acc[1] += cxz;
          acc[2] += cxz;
          acc[3] += czz;
        } else {
          // C = w rho [vp n n^T + vs (I - n n^T)] in (r, theta, z), theta
          // tangential, transformed to (U+, U-, U_z): T C T^T.
          const double ct = w * mat.rho * mat.vs;
          const double cpp = 0.5 * (cxx + ct), cpm = 0.5 * (cxx - ct), cpz = kSqrtHalf * cxz;
          const double c[9] = {cpp, cpm, cpz, cpm, cpp, cpz, cpz, cpz, czz};
          for (int s = 0; s < 9; ++s) acc[s] += c[s];
        }
      }
    }
  }
  const std::set<index_t> axis(d.axis_points.begin(), d.axis_points.end());
  for (auto [g, c] : absorbing) {
    if (dirichlet.count(g)) continue;  // Dirichlet wins at shared corners
    if (axis.count(g)) {
      // Components fixed on the axis are decoupled from the damping.
      for (int ac : d.axis_components) {
        for (int s = 0; s < nc; ++s) c[ac * nc + s] = c[s * nc + ac] = 0.0;
      }
    }
    d.absorbing_points.push_back(g);
    for (int s = 0; s < nc * nc; ++s) d.absorbing_coef.push_back(static_cast<real_t>(c[s]));
  }
  d.dirichlet_points.assign(dirichlet.begin(), dirichlet.end());
}

void Solver::face_weight(const Problem& p, index_t leaf, int f, int i, int j, double& w, double& nx,
                         double& nz) const {
  const int ngll = p.gll.size();
  const std::size_t pt = static_cast<std::size_t>(leaf) * ngll * ngll + j * ngll + i;
  double ds;
  face_measure(p.geometry, pt, f, 1.0, ds, nx, nz);
  const bool along_xi = f == 0 || f == 2;
  if (!axisym_) {
    w = p.gll.weights[along_xi ? i : j] * ds;
  } else if (along_xi && p.forest.is_axial(leaf)) {
    // GLJ rule along a radial face starting on the axis (see build_domains).
    const double x_xi = p.geometry.eta_z[pt] * p.geometry.jac[pt];
    const double rt = i == 0 ? x_xi : p.geometry.x[pt] / (1.0 + glj_.points[i]);
    w = glj_.weights[i] * ds * rt;
  } else {
    w = p.gll.weights[along_xi ? i : j] * ds * std::max(p.geometry.x[pt], 0.0);
  }
}

void Solver::build_coupling(const Problem& p) {
  const int fi = domain_index(Medium::Acoustic);
  const int si = domain_index(Medium::Elastic);
  if (fi < 0 || si < 0) return;
  const Domain& fluid = domains_[fi];
  const Domain& solid = domains_[si];
  const int ngll = p.gll.size();
  const int n = ngll - 1;
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
        face_weight(p, leaf, f, i, j, w, nx, nz);
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
    const bool axial = axisym_ && p.forest.is_axial(loc->leaf);
    const PointWeights w = point_weights(p.forest, axial ? lagrange_glj_ : lagrange_, lagrange_, *loc);
    const index_t e = d.element_of_leaf[loc->leaf];
    const int npe = d.dofs.points_per_element();
    // Axisymmetric: the weak form is per radian (azimuthal order 0, the
    // theta integral of 1) or per pi (order 1, the integral of cos^2), so
    // source strengths are divided by 2 pi or pi. On the axis this is a 3D
    // point source of the given strength; off the axis a ring source of that
    // net strength (docs/DESIGN.md §2.4, §2.5).
    const double scale = !axisym_ ? 1.0 : azimuthal_order_ == 1 ? 1.0 / kPi : 1.0 / (2.0 * kPi);
    const double rs = src.position[0];
    const auto& c = src.components;
    const bool directional = src.kind == Source::Kind::Force || src.kind == Source::Kind::Dipole;
    if (axisym_) {
      const bool on_axis = std::abs(rs) <= 1e-12 * (1.0 + std::abs(src.position[1]));
      if (azimuthal_order_ == 0) {
        INSEM_CHECK(!(on_axis && directional && c[0] != 0.0),
                    "source " << si << ": a radial force or dipole on the axis is not axisymmetric (it is "
                              << "azimuthal order 1)");
        INSEM_CHECK(!(on_axis && src.kind == Source::Kind::MomentTensor && c[2] != 0.0),
                    "source " << si << ": an M_rz moment on the axis is not axisymmetric");
      } else {
        INSEM_CHECK(directional, "source " << si << " (" << src.kind_name()
                                           << "): azimuthal order 1 takes force and dipole sources only");
        INSEM_CHECK(!(on_axis && c[1] != 0.0),
                    "source " << si << ": a vertical force or dipole on the axis is axisymmetric (azimuthal "
                              << "order 0)");
      }
    }
    const double rho = material_of_leaf(p, loc->leaf).rho;
    const bool pm = d.medium == Medium::Elastic && d.azimuthal_order == 1;

    // Force on each local point, then distributed to global points through
    // the slot's constraint row (C^T), accumulated in global order.
    std::map<index_t, std::array<double, 3>> acc;
    for (int k = 0; k < npe; ++k) {
      double fc[3] = {0.0, 0.0, 0.0};
      switch (src.kind) {
        case Source::Kind::Force:
          if (pm) {
            // f_x W_r = f_x (W+ + W-) / sqrt 2, f_z W_z.
            fc[0] = fc[1] = kSqrtHalf * c[0] * w.value[k];
            fc[2] = c[1] * w.value[k];
          } else {
            fc[0] = c[0] * w.value[k];
            fc[1] = c[1] * w.value[k];
          }
          break;
        case Source::Kind::MomentTensor:
          // Equivalent body force -div(M delta): f_a(p) = M_ab d_b phi_p(x_s).
          fc[0] = c[0] * w.d_dx[k] + c[2] * w.d_dz[k];
          fc[1] = c[2] * w.d_dx[k] + c[1] * w.d_dz[k];
          if (axisym_) {
            // Hoop part M_tt eps_tt(phi), with M_tt = M_rr: phi / r, or
            // dphi/dr on the axis.
            fc[0] += c[0] * (rs > 0.0 ? w.value[k] / rs : w.d_dx[k]);
          }
          break;
        case Source::Kind::Monopole:
          fc[0] = w.value[k];
          break;
        case Source::Kind::Dipole:
          // (1/kappa) p_tt = div(grad p / rho) - div(f delta / rho): the
          // weak form gets f . grad phi(x_s) / rho.
          fc[0] = (c[0] * w.d_dx[k] + c[1] * w.d_dz[k]) / rho;
          break;
      }
      for (const auto& [g, wt] : d.dofs.slot_entries(e, k)) {
        for (int comp = 0; comp < d.ncomp; ++comp) acc[g][comp] += wt * fc[comp] * scale;
      }
    }
    SourceTerm term;
    term.domain = di;
    term.stf = src.stf;
    for (const auto& [g, f] : acc) term.points.push_back(g);
    term.coef.resize(static_cast<std::size_t>(d.ncomp) * acc.size());
    std::size_t i = 0;
    for (const auto& [g, f] : acc) {
      for (int comp = 0; comp < d.ncomp; ++comp) term.coef[comp * acc.size() + i] = static_cast<real_t>(f[comp]);
      ++i;
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
    const bool axial = axisym_ && p.forest.is_axial(loc->leaf);
    const PointWeights w = point_weights(p.forest, axial ? lagrange_glj_ : lagrange_, lagrange_, *loc);
    const index_t e = d.element_of_leaf[loc->leaf];
    // Interpolation weights on global points (constrained slots expanded).
    std::map<index_t, std::array<double, 3>> acc;
    for (int k = 0; k < d.dofs.points_per_element(); ++k) {
      for (const auto& [g, wt] : d.dofs.slot_entries(e, k)) {
        acc[g][0] += wt * w.value[k];
        acc[g][1] += wt * w.d_dx[k];
        acc[g][2] += wt * w.d_dz[k];
      }
    }
    for (const auto& [g, c] : acc) {
      term.points.push_back(g);
      term.value.push_back(static_cast<real_t>(c[0]));
      term.d_dx.push_back(static_cast<real_t>(c[1]));
      term.d_dz.push_back(static_cast<real_t>(c[2]));
    }
    term.rho = material_of_leaf(p, loc->leaf).rho;
    INSEM_CHECK(!rec.quantities.empty(), "receiver '" << rec.name << "' records no quantities");
    for (const std::string& q : rec.quantities) {
      Trace t;
      t.receiver = rec.name;
      t.quantity = q;
      t.ncomp = quantity_components(q, d.medium, d.azimuthal_order);
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
      d.apply_dirichlet(x);
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
    for (Domain* d : {&solid, &fluid}) d->apply_dirichlet(d->u);
    const double nx = mass_norm2(solid, solid.u) + mass_norm2(fluid, fluid.u);
    for (Domain* d : {&solid, &fluid}) {
      internal_forces(*d, d->u, false);
      add_coupling_forces(*d);
      const std::size_t ng = static_cast<std::size_t>(d->n_global());
      for (int c = 0; c < d->ncomp; ++c) {
        for (std::size_t g = 0; g < ng; ++g) d->a[c * ng + g] = d->f[c * ng + g] * d->inv_mass[g];
      }
      d->apply_dirichlet(d->a);
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
  // Azimuthal order 1: the radial part goes to (or comes from) U+ and U-
  // alike, U_r = (U+ + U-) / sqrt 2; the theta component takes no traction.
  const bool pm = d.azimuthal_order == 1;
  const real_t h = static_cast<real_t>(kSqrtHalf);
  if (d.medium == Medium::Elastic) {
    // Traction p n_f on the solid.
    const Domain& fluid = domains_[domain_index(Medium::Acoustic)];
    const std::size_t ng = static_cast<std::size_t>(d.n_global());
    const std::size_t zc = pm ? 2 * ng : ng;
    for (std::size_t i = 0; i < n; ++i) {
      const real_t p = fluid.u[coupling_.fluid_points[i]];
      const std::size_t g = coupling_.solid_points[i];
      if (pm) {
        d.f[g] += h * coupling_.cx[i] * p;
        d.f[ng + g] += h * coupling_.cx[i] * p;
      } else {
        d.f[g] += coupling_.cx[i] * p;
      }
      d.f[zc + g] += coupling_.cz[i] * p;
    }
  } else {
    // -n_f . a_s on the fluid.
    const Domain& solid = domains_[domain_index(Medium::Elastic)];
    const std::size_t ngs = static_cast<std::size_t>(solid.n_global());
    const std::size_t zc = pm ? 2 * ngs : ngs;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t gs = coupling_.solid_points[i];
      const real_t ar = pm ? h * (solid.a[gs] + solid.a[ngs + gs]) : solid.a[gs];
      d.f[coupling_.fluid_points[i]] -= coupling_.cx[i] * ar + coupling_.cz[i] * solid.a[zc + gs];
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
  // i.e. (M + dt/2 C) a = f - C v^{n+1/2}, solved per point (C is
  // ncomp x ncomp). Treating the damping implicitly keeps it unconditionally
  // stable; at t = 0 the velocity is known and the term is explicit.
  const double h = damping_implicit ? 0.5 * dt_ : 0.0;
  const std::size_t nabs = d.absorbing_points.size();
  const std::size_t nc2 = static_cast<std::size_t>(ncomp) * ncomp;
  for (std::size_t i = 0; i < nabs; ++i) {
    const std::size_t g = d.absorbing_points[i];
    const real_t* c = &d.absorbing_coef[nc2 * i];
    double A[9], rhs[3];
    for (int r = 0; r < ncomp; ++r) {
      rhs[r] = d.f[r * ng + g];
      for (int s = 0; s < ncomp; ++s) {
        rhs[r] -= static_cast<double>(c[r * ncomp + s]) * d.v[s * ng + g];
        A[r * ncomp + s] = h * c[r * ncomp + s] + (r == s ? static_cast<double>(d.mass[g]) : 0.0);
      }
    }
    solve_spd(ncomp, A, rhs);
    for (int r = 0; r < ncomp; ++r) d.a[r * ng + g] = static_cast<real_t>(rhs[r]);
  }
  d.apply_dirichlet(d.a);
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
  d.apply_dirichlet(d.u);
  d.apply_dirichlet(d.v);
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
      if (d.medium == Medium::Elastic && d.azimuthal_order == 1) {
        // (U+, U-) -> (U_r, U_theta)
        const double up = interpolate(field, 0), um = interpolate(field, 1);
        t.data.push_back(kSqrtHalf * (up + um));
        t.data.push_back(kSqrtHalf * (up - um));
        t.data.push_back(interpolate(field, 2));
        continue;
      }
      for (int c = 0; c < t.ncomp; ++c) t.data.push_back(interpolate(field, c));
    }
  }
}

void Solver::record_snapshot() {
  snapshot_times_.push_back(time());
  for (Snapshot& snap : snapshots_) {
    Domain& d = domains_[domain_index(snap.medium)];
    const std::vector<real_t>& field = (snap.quantity == "velocity") ? d.v : (snap.quantity == "acceleration") ? d.a : d.u;
    // d.u_local is kernel scratch, free between steps.
    d.dofs.gather(field.data(), d.u_local.data(), d.ncomp);
    const std::size_t npe = static_cast<std::size_t>(d.dofs.points_per_element());
    const std::size_t base = snap.data.size();
    for (int c = 0; c < d.ncomp; ++c) {
      for (std::size_t e = 0; e < d.leaves.size(); ++e) {
        const real_t* src = &d.u_local[(e * d.ncomp + c) * npe];
        snap.data.insert(snap.data.end(), src, src + npe);
      }
    }
    if (d.medium == Medium::Elastic && d.azimuthal_order == 1) {
      // (U+, U-) -> (U_r, U_theta)
      const std::size_t n = d.leaves.size() * npe;
      float* x = &snap.data[base];
      for (std::size_t i = 0; i < n; ++i) {
        const double up = x[i], um = x[n + i];
        x[i] = static_cast<float>(kSqrtHalf * (up + um));
        x[n + i] = static_cast<float>(kSqrtHalf * (up - um));
      }
    }
  }
}

std::vector<double> Solver::snapshot_points(Medium medium) const {
  const int di = domain_index(medium);
  INSEM_CHECK(di >= 0, "the model has no " << to_string(medium) << " region");
  const Domain& d = domains_[di];
  const std::size_t npe = static_cast<std::size_t>(d.dofs.points_per_element());
  const std::size_t n = d.leaves.size() * npe;
  std::vector<double> out(2 * n);
  for (std::size_t e = 0; e < d.leaves.size(); ++e) {
    for (std::size_t k = 0; k < npe; ++k) {
      const std::size_t q = static_cast<std::size_t>(d.leaves[e]) * npe + k;
      out[e * npe + k] = geometry_x_[q];
      out[n + e * npe + k] = geometry_z_[q];
    }
  }
  return out;
}

std::vector<std::int32_t> Solver::snapshot_cells(Medium medium) const {
  const int di = domain_index(medium);
  INSEM_CHECK(di >= 0, "the model has no " << to_string(medium) << " region");
  const Domain& d = domains_[di];
  const int ngll = d.dofs.ngll();
  const std::int32_t npe = ngll * ngll;
  std::vector<std::int32_t> out;
  out.reserve(d.leaves.size() * (ngll - 1) * (ngll - 1) * 4);
  for (std::size_t e = 0; e < d.leaves.size(); ++e) {
    const std::int32_t base = static_cast<std::int32_t>(e) * npe;
    for (int j = 0; j + 1 < ngll; ++j) {
      for (int i = 0; i + 1 < ngll; ++i) {
        out.insert(out.end(), {base + j * ngll + i, base + j * ngll + i + 1, base + (j + 1) * ngll + i + 1,
                               base + (j + 1) * ngll + i});
      }
    }
  }
  return out;
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
  if (snapshot_every_ > 0) record_snapshot();
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
    if (snapshot_every_ > 0 && step_ % snapshot_every_ == 0) record_snapshot();
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
  f.set_attribute("/", "coords", std::string(axisym_ ? "axisymmetric" : "cartesian"));
  f.set_attribute("/", "azimuthal_order", static_cast<std::int64_t>(azimuthal_order_));
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
  if (!snapshots_.empty()) {
    f.write("/snapshots/time", snapshot_times_);
    const std::size_t ns = snapshot_times_.size();
    for (Medium medium : {Medium::Acoustic, Medium::Elastic}) {
      if (domain_index(medium) < 0) continue;
      const std::string g = "/snapshots/" + to_string(medium);
      const auto pts = snapshot_points(medium);
      const std::size_t n = pts.size() / 2;
      std::vector<double> xz(2 * n);
      for (std::size_t i = 0; i < n; ++i) {
        xz[2 * i] = pts[i];
        xz[2 * i + 1] = pts[n + i];
      }
      f.write(g + "/points", xz.data(), {n, 2});
      const auto cells = snapshot_cells(medium);
      f.write(g + "/cells", cells.data(), {cells.size() / 4, 4});
    }
    for (const Snapshot& snap : snapshots_) {
      const std::size_t n = snap.data.size() / (ns * snap.ncomp);
      f.write("/snapshots/" + to_string(snap.medium) + "/" + snap.quantity, snap.data.data(),
              {ns, static_cast<std::size_t>(snap.ncomp), n});
    }
  }
  if (!energy_.empty()) {
    std::vector<double> e;
    for (const auto& row : energy_) e.insert(e.end(), row.begin(), row.end());
    f.write("/energy", e.data(), {energy_.size(), 3});
    f.set_attribute("/energy", "columns", std::string("time,kinetic,potential"));
  }
}

}  // namespace insem::solver
