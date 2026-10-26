// Python bindings: the C++ object model exposed as insem._core. The pure
// Python layer in python/insem builds on these.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "insem/basis/lagrange.hpp"
#include "insem/basis/quadrature.hpp"
#include "insem/core/config.hpp"
#include "insem/core/error.hpp"
#include "insem/geometry/geometry.hpp"
#include "insem/mesh/forest.hpp"
#include "insem/mesh/macro_mesh.hpp"
#include "insem/ops/dof_map.hpp"
#include "insem/sim/description.hpp"
#include "insem/sim/simulation.hpp"
#include "insem/solver/solver.hpp"

namespace py = pybind11;
using namespace insem;

namespace {

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
py::array_t<T> to_numpy(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
  py::array_t<T> out(shape);
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

template <class T>
py::array_t<T> to_numpy(const std::vector<T>& v) {
  return to_numpy(v, {static_cast<py::ssize_t>(v.size())});
}

template <class T>
std::vector<T> to_vector(const CArray<T>& a) {
  return std::vector<T>(a.data(), a.data() + a.size());
}

py::tuple rule_to_tuple(const basis::Rule& r) {
  return py::make_tuple(to_numpy(r.points), to_numpy(r.weights));
}

std::shared_ptr<mesh::MacroMesh> make_macro_mesh(const std::string& cell_type,
                                                 const std::string& coords,
                                                 const CArray<double>& nodes,
                                                 const CArray<std::int32_t>& cells,
                                                 const CArray<std::int32_t>& material,
                                                 const CArray<std::int32_t>& boundary) {
  const auto type = mesh::cell_type_from_string(cell_type);
  INSEM_CHECK(nodes.ndim() == 2 && nodes.shape(1) == 2, "nodes must have shape (n_nodes, 2)");
  INSEM_CHECK(cells.ndim() == 2 && cells.shape(1) == static_cast<int>(type),
              "cells must have shape (n_cells, " << static_cast<int>(type) << ") for " << cell_type);
  INSEM_CHECK(material.ndim() == 1 && material.shape(0) == cells.shape(0),
              "material must have shape (n_cells,)");
  INSEM_CHECK(boundary.size() == 0 || (boundary.ndim() == 2 && boundary.shape(1) == 3),
              "boundary must have shape (n_faces, 3): (cell, face, tag)");
  std::vector<mesh::BoundaryFace> faces;
  const auto* b = boundary.data();
  for (py::ssize_t i = 0; i < boundary.size() / 3; ++i) faces.push_back({b[3 * i], b[3 * i + 1], b[3 * i + 2]});
  auto m = to_vector(material);
  return std::make_shared<mesh::MacroMesh>(type, mesh::coord_system_from_string(coords),
                                           to_vector(nodes), to_vector(cells),
                                           std::vector<int>(m.begin(), m.end()), std::move(faces));
}

// Global field (n_global,) or (ncomp, n_global) -> local (n_elem, [ncomp,] ngll, ngll).
py::array_t<double> gather(const ops::DofMap& d, const CArray<double>& global) {
  const bool scalar = global.ndim() == 1;
  INSEM_CHECK((scalar && global.shape(0) == d.n_global()) ||
                  (global.ndim() == 2 && global.shape(1) == d.n_global()),
              "global field must have shape (n_global,) or (ncomp, n_global)");
  const int ncomp = scalar ? 1 : static_cast<int>(global.shape(0));
  std::vector<py::ssize_t> shape{d.n_elements()};
  if (!scalar) shape.push_back(ncomp);
  shape.push_back(d.ngll());
  shape.push_back(d.ngll());
  py::array_t<double> local(shape);
  d.gather(global.data(), local.mutable_data(), ncomp);
  return local;
}

py::array_t<double> assemble(const ops::DofMap& d, const CArray<double>& local) {
  const bool scalar = local.ndim() == 3;
  INSEM_CHECK((scalar || local.ndim() == 4) && local.shape(0) == d.n_elements() &&
                  local.shape(local.ndim() - 1) == d.ngll() && local.shape(local.ndim() - 2) == d.ngll(),
              "local field must have shape (n_elements, [ncomp,] ngll, ngll)");
  const int ncomp = scalar ? 1 : static_cast<int>(local.shape(1));
  std::vector<py::ssize_t> shape;
  if (!scalar) shape.push_back(ncomp);
  shape.push_back(d.n_global());
  py::array_t<double> global(shape);
  d.assemble(local.data(), global.mutable_data(), ncomp);
  return global;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "insem C++ core";
  m.attr("__version__") = insem::version();
  m.attr("real_dtype") = INSEM_REAL_IS_DOUBLE ? "float64" : "float32";
  py::register_exception<insem::Error>(m, "InsemError", PyExc_RuntimeError);

  // ---- basis
  m.def("gll", [](int n) { return rule_to_tuple(basis::gll(n)); }, py::arg("n_points"),
        "Gauss-Lobatto-Legendre points and weights on [-1, 1].");
  m.def("glj01", [](int n) { return rule_to_tuple(basis::glj01(n)); }, py::arg("n_points"),
        "Gauss-Lobatto-Jacobi (0,1) points and weights (weight 1 + x) on [-1, 1].");
  m.def("gauss_lobatto_jacobi",
        [](int n, int a, int b) { return rule_to_tuple(basis::gauss_lobatto_jacobi(n, a, b)); },
        py::arg("n_points"), py::arg("alpha"), py::arg("beta"));
  m.def("jacobi", &basis::jacobi, py::arg("n"), py::arg("a"), py::arg("b"), py::arg("x"));
  m.def(
      "derivative_matrix",
      [](const CArray<double>& nodes) {
        const basis::LagrangeBasis b(to_vector(nodes));
        const auto n = static_cast<py::ssize_t>(b.size());
        return to_numpy(b.derivative_matrix(), {n, n});
      },
      py::arg("nodes"), "D[i, j] = l_j'(x_i) for the Lagrange basis on `nodes`.");
  m.def(
      "interpolation_matrix",
      [](const CArray<double>& nodes, const CArray<double>& targets) {
        const basis::LagrangeBasis b(to_vector(nodes));
        return to_numpy(b.interpolation_matrix(to_vector(targets)),
                        {static_cast<py::ssize_t>(targets.size()), static_cast<py::ssize_t>(b.size())});
      },
      py::arg("nodes"), py::arg("targets"), "I[t, j] = l_j(targets[t]).");

  // ---- mesh
  py::class_<mesh::MacroMesh, std::shared_ptr<mesh::MacroMesh>>(m, "MacroMesh")
      .def(py::init(&make_macro_mesh), py::arg("cell_type"), py::arg("coords"), py::arg("nodes"),
           py::arg("cells"), py::arg("material"),
           py::arg("boundary") = CArray<std::int32_t>(std::vector<py::ssize_t>{0, 3}))
      .def_property_readonly("cell_type", [](const mesh::MacroMesh& mm) { return mesh::to_string(mm.cell_type()); })
      .def_property_readonly("coords", [](const mesh::MacroMesh& mm) { return mesh::to_string(mm.coord_system()); })
      .def_property_readonly("n_nodes", &mesh::MacroMesh::n_nodes)
      .def_property_readonly("n_cells", &mesh::MacroMesh::n_cells)
      .def_property_readonly("n_edges", &mesh::MacroMesh::n_edges)
      .def_property_readonly("nodes", [](const mesh::MacroMesh& mm) {
        return to_numpy(mm.nodes(), {mm.n_nodes(), 2});
      })
      .def_property_readonly("cells", [](const mesh::MacroMesh& mm) {
        return to_numpy(mm.cells(), {mm.n_cells(), mm.nodes_per_cell()});
      })
      .def_property_readonly("material", [](const mesh::MacroMesh& mm) {
        std::vector<std::int32_t> v(mm.materials().begin(), mm.materials().end());
        return to_numpy(v);
      })
      .def_property_readonly("boundary", [](const mesh::MacroMesh& mm) {
        std::vector<std::int32_t> v;
        for (const auto& f : mm.boundary()) v.insert(v.end(), {f.cell, f.face, f.tag});
        return to_numpy(v, {static_cast<py::ssize_t>(mm.boundary().size()), 3});
      })
      .def("map", &mesh::MacroMesh::map, py::arg("cell"), py::arg("xi"), py::arg("eta"))
      .def("jacobian", &mesh::MacroMesh::jacobian, py::arg("cell"), py::arg("xi"), py::arg("eta"))
      .def("__repr__", [](const mesh::MacroMesh& mm) {
        return "<MacroMesh " + std::to_string(mm.n_cells()) + " " + mesh::to_string(mm.cell_type()) +
               " cells, " + mesh::to_string(mm.coord_system()) + ">";
      });

  py::class_<mesh::Forest>(m, "Forest")
      .def_static(
          "uniform",
          [](std::shared_ptr<mesh::MacroMesh> macro, int level) {
            return mesh::Forest::uniform(std::move(macro), level);
          },
          py::arg("macro"), py::arg("level"))
      .def_property_readonly("n_leaves", &mesh::Forest::n_leaves)
      .def_property_readonly("leaves", [](const mesh::Forest& f) {
        std::vector<std::int32_t> v;
        v.reserve(4 * f.leaves().size());
        for (const auto& l : f.leaves()) v.insert(v.end(), {l.tree, l.level, l.i, l.j});
        return to_numpy(v, {f.n_leaves(), 4});
      }, "(n_leaves, 4) array of (tree, level, i, j).")
      .def("leaf_box", &mesh::Forest::leaf_box, py::arg("leaf"))
      .def_property_readonly("max_level", &mesh::Forest::max_level)
      .def("face_neighbor", [](const mesh::Forest& f, index_t l, int face) {
        const auto nb = f.face_neighbor(l, face);
        return py::make_tuple(nb.leaf, nb.face, nb.level_diff);
      }, py::arg("leaf"), py::arg("face"), "(neighbour leaf or -1, its face, level difference).")
      .def("leaf_corners", [](const mesh::Forest& f, index_t l) {
        std::vector<double> out;
        for (auto [xi, eta] : {std::pair{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}}) {
          const auto p = geometry::leaf_map(f, l, xi, eta);
          out.insert(out.end(), p.begin(), p.end());
        }
        return to_numpy(out, {4, 2});
      }, py::arg("leaf"), "(4, 2) physical corner coordinates of a leaf.");

  // ---- ops
  py::class_<ops::DofMap>(m, "DofMap")
      .def_static("build", py::overload_cast<const mesh::Forest&, int>(&ops::DofMap::build), py::arg("forest"),
                  py::arg("ngll"))
      .def_property_readonly("ngll", &ops::DofMap::ngll)
      .def_property_readonly("n_elements", &ops::DofMap::n_elements)
      .def_property_readonly("n_global", &ops::DofMap::n_global)
      .def_property_readonly("local_to_global", [](const ops::DofMap& d) {
        return to_numpy(d.local_to_global(), {d.n_elements(), d.ngll(), d.ngll()});
      }, "(n_elements, ngll, ngll) global index of each local point, [e, j, i].")
      .def_property_readonly("multiplicity", [](const ops::DofMap& d) { return to_numpy(d.multiplicity()); })
      .def("gather", &gather, py::arg("global_field"))
      .def("assemble", &assemble, py::arg("local_field"));

  // ---- geometry
  py::class_<geometry::Geometry>(m, "Geometry")
      .def_readonly("ngll", &geometry::Geometry::ngll)
      .def_readonly("n_elements", &geometry::Geometry::n_elements)
      .def_property_readonly("kind", [](const geometry::Geometry& g) {
        std::vector<std::string> v;
        for (auto k : g.kind) v.push_back(geometry::to_string(k));
        return v;
      })
#define INSEM_GEOMETRY_ARRAY(name)                                                        \
  .def_property_readonly(#name, [](const geometry::Geometry& g) {                         \
    return to_numpy(g.name, {g.n_elements, g.ngll, g.ngll});                              \
  })
      INSEM_GEOMETRY_ARRAY(x) INSEM_GEOMETRY_ARRAY(z) INSEM_GEOMETRY_ARRAY(jac)
      INSEM_GEOMETRY_ARRAY(xi_x) INSEM_GEOMETRY_ARRAY(xi_z)
      INSEM_GEOMETRY_ARRAY(eta_x) INSEM_GEOMETRY_ARRAY(eta_z);
#undef INSEM_GEOMETRY_ARRAY

  m.def(
      "global_coordinates",
      [](const ops::DofMap& d, const geometry::Geometry& g) {
        const auto gc = geometry::global_coordinates(d, g);
        std::vector<double> xz(gc.x);
        xz.insert(xz.end(), gc.z.begin(), gc.z.end());
        return py::make_tuple(to_numpy(xz, {2, d.n_global()}), gc.max_mismatch);
      },
      py::arg("dofs"), py::arg("geometry"),
      "(coordinates (2, n_global), max mismatch between element-local copies).");

  // ---- problem description
  py::class_<Material>(m, "Material")
      .def_static("acoustic", [](double vp, double rho) {
        const Material mat = Material::acoustic(vp, rho);
        mat.validate();
        return mat;
      }, py::arg("vp"), py::arg("rho"))
      .def_static("elastic", [](double vp, double vs, double rho) {
        const Material mat = Material::elastic(vp, vs, rho);
        mat.validate();
        return mat;
      }, py::arg("vp"), py::arg("vs"), py::arg("rho"))
      .def_property_readonly("medium", [](const Material& mat) { return to_string(mat.medium); })
      .def_readonly("vp", &Material::vp)
      .def_readonly("vs", &Material::vs)
      .def_readonly("rho", &Material::rho)
      .def("__repr__", [](const Material& mat) {
        std::ostringstream os;
        os << "<Material " << to_string(mat.medium) << " vp=" << mat.vp;
        if (mat.medium == Medium::Elastic) os << " vs=" << mat.vs;
        os << " rho=" << mat.rho << ">";
        return os.str();
      });

  py::class_<SourceTimeFunction>(m, "SourceTimeFunction")
      .def_property_readonly("kind", &SourceTimeFunction::kind_name)
      .def_readonly("f0", &SourceTimeFunction::f0)
      .def_readonly("t0", &SourceTimeFunction::t0)
      .def_readonly("amplitude", &SourceTimeFunction::amplitude)
      .def("__call__", [](const SourceTimeFunction& stf, double t) { return stf(t); }, py::arg("t"))
      .def("__call__", [](const SourceTimeFunction& stf, const CArray<double>& t) {
        py::array_t<double> out(t.size());
        for (py::ssize_t i = 0; i < t.size(); ++i) out.mutable_data()[i] = stf(t.data()[i]);
        return out;
      }, py::arg("t"));
  auto t0_or_default = [](py::object t0) { return t0.is_none() ? -1.0 : t0.cast<double>(); };
  m.def("ricker", [t0_or_default](double f0, py::object t0, double amplitude) {
          return SourceTimeFunction::ricker(f0, t0_or_default(t0), amplitude);
        }, py::arg("f0"), py::arg("t0") = py::none(), py::arg("amplitude") = 1.0,
        "Ricker wavelet with dominant frequency f0 (default delay t0 = 1.2 / f0).");
  m.def("gaussian", [t0_or_default](double f0, py::object t0, double amplitude) {
          return SourceTimeFunction::gaussian(f0, t0_or_default(t0), amplitude);
        }, py::arg("f0"), py::arg("t0") = py::none(), py::arg("amplitude") = 1.0);
  m.def("gaussian_derivative", [t0_or_default](double f0, py::object t0, double amplitude) {
          return SourceTimeFunction::gaussian_derivative(f0, t0_or_default(t0), amplitude);
        }, py::arg("f0"), py::arg("t0") = py::none(), py::arg("amplitude") = 1.0);
  m.def("sampled_stf", [](const CArray<double>& samples, double dt, double amplitude) {
          return SourceTimeFunction::sampled(to_vector(samples), dt, amplitude);
        }, py::arg("samples"), py::arg("dt"), py::arg("amplitude") = 1.0,
        "Source time function from samples at t = i * dt (linear interpolation).");

  py::class_<Source>(m, "Source")
      .def_static("force", &Source::force, py::arg("x"), py::arg("z"), py::arg("fx"), py::arg("fz"),
                  py::arg("stf"))
      .def_static("moment_tensor", &Source::moment_tensor, py::arg("x"), py::arg("z"), py::arg("mxx"),
                  py::arg("mzz"), py::arg("mxz"), py::arg("stf"))
      .def_static("monopole", &Source::monopole, py::arg("x"), py::arg("z"), py::arg("stf"),
                  "Acoustic volume-injection source; stf is d^2Q/dt^2 (m^3/s^2 per unit length in 2D).")
      .def_static("dipole", &Source::dipole, py::arg("x"), py::arg("z"), py::arg("fx"), py::arg("fz"),
                  py::arg("stf"),
                  "Acoustic point force (pressure dipole) of direction (fx, fz) in N; stf is the force time "
                  "function. In axisymmetric models a horizontal dipole on the axis needs azimuthal_order=1.")
      .def_property_readonly("kind", &Source::kind_name)
      .def_readonly("position", &Source::position)
      .def_readonly("components", &Source::components)
      .def_readonly("stf", &Source::stf);

  py::class_<Receiver>(m, "Receiver")
      .def_readonly("name", &Receiver::name)
      .def_readonly("position", &Receiver::position)
      .def_readonly("quantities", &Receiver::quantities)
      .def_readonly("medium", &Receiver::medium);

  py::class_<TimeConfig>(m, "TimeConfig")
      .def_readonly("dt", &TimeConfig::dt)
      .def_readonly("safety", &TimeConfig::safety)
      .def_readonly("n_steps", &TimeConfig::n_steps)
      .def_readonly("duration", &TimeConfig::duration)
      .def_readonly("record_every", &TimeConfig::record_every)
      .def_readonly("energy_every", &TimeConfig::energy_every)
      .def_readonly("snapshot_every", &TimeConfig::snapshot_every)
      .def_readonly("snapshot_quantities", &TimeConfig::snapshot_quantities);

  // ---- solver
  py::class_<solver::Solver>(m, "Solver")
      .def_property_readonly("dt", &solver::Solver::dt)
      .def_property_readonly("stable_dt", &solver::Solver::stable_dt)
      .def_property_readonly("n_steps", &solver::Solver::n_steps)
      .def_property_readonly("step", &solver::Solver::step)
      .def_property_readonly("time", &solver::Solver::time)
      .def("advance", &solver::Solver::advance, py::arg("n_steps"), py::call_guard<py::gil_scoped_release>())
      .def("run", &solver::Solver::run, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("media", [](const solver::Solver& s) {
        std::vector<std::string> out;
        for (std::size_t i = 0; i < s.n_domains(); ++i) out.push_back(to_string(s.domain(i).medium));
        return out;
      })
      .def("coordinates", [](const solver::Solver& s, const std::string& medium) {
        const int di = s.domain_index(medium_from_string(medium));
        INSEM_CHECK(di >= 0, "the model has no " << medium << " region");
        const auto& d = s.domain(di);
        return to_numpy(d.coords, {2, d.n_global()});
      }, py::arg("medium"), "(2, n_points) coordinates of the global points of a medium.")
      .def("state", [](const solver::Solver& s, const std::string& medium) {
        const int di = s.domain_index(medium_from_string(medium));
        INSEM_CHECK(di >= 0, "the model has no " << medium << " region");
        const auto& d = s.domain(di);
        auto conv = [&](const std::vector<real_t>& v) {
          return to_numpy(std::vector<double>(v.begin(), v.end()), {d.ncomp, d.n_global()});
        };
        return py::make_tuple(conv(d.u), conv(d.v), conv(d.a));
      }, py::arg("medium"),
           "(field, first, second time derivative), each (ncomp, n_points): displacement (elastic) or pressure "
           "(acoustic). Azimuthal order 1 elastic components are (U+, U-, U_z), U+- = (U_r +- U_theta) / sqrt 2.")
      .def("set_state", [](solver::Solver& s, const std::string& medium, const CArray<double>& u,
                           const CArray<double>& v) {
        s.set_state(medium_from_string(medium), to_vector(u), to_vector(v));
      }, py::arg("medium"), py::arg("u"), py::arg("v"))
      .def("trace", [](const solver::Solver& s, const std::string& receiver, const std::string& quantity) {
        for (const auto& t : s.traces()) {
          if (t.receiver == receiver && t.quantity == quantity) {
            return to_numpy(t.data, {static_cast<py::ssize_t>(t.data.size() / t.ncomp), t.ncomp});
          }
        }
        throw Error("no trace '" + quantity + "' for receiver '" + receiver + "'");
      }, py::arg("receiver"), py::arg("quantity"), "(n_samples, ncomp) recorded trace.")
      .def_property_readonly("times", [](const solver::Solver& s) { return to_numpy(s.sample_times()); })
      .def_property_readonly("energy", [](const solver::Solver& s) {
        std::vector<double> e;
        for (const auto& row : s.energy()) e.insert(e.end(), row.begin(), row.end());
        return to_numpy(e, {static_cast<py::ssize_t>(s.energy().size()), 3});
      }, "(n, 3) energy monitor samples: time, kinetic, potential.")
      .def("snapshot", [](const solver::Solver& s, const std::string& medium, const std::string& quantity) {
        const Medium m = medium_from_string(medium);
        for (const auto& snap : s.snapshots()) {
          if (snap.medium != m || snap.quantity != quantity) continue;
          const py::ssize_t ns = static_cast<py::ssize_t>(s.snapshot_times().size());
          const py::ssize_t n = ns ? static_cast<py::ssize_t>(snap.data.size()) / (ns * snap.ncomp) : 0;
          return to_numpy(snap.data, {ns, snap.ncomp, n});
        }
        throw Error("no " + quantity + " snapshots for the " + medium + " medium");
      }, py::arg("medium"), py::arg("quantity"), "(n_snapshots, ncomp, n_points) snapshot data.")
      .def("snapshot_points", [](const solver::Solver& s, const std::string& medium) {
        const auto p = s.snapshot_points(medium_from_string(medium));
        return to_numpy(p, {2, static_cast<py::ssize_t>(p.size() / 2)});
      }, py::arg("medium"), "(2, n_points) coordinates of the snapshot points.")
      .def("snapshot_cells", [](const solver::Solver& s, const std::string& medium) {
        const auto c = s.snapshot_cells(medium_from_string(medium));
        return to_numpy(c, {static_cast<py::ssize_t>(c.size() / 4), 4});
      }, py::arg("medium"), "(n_cells, 4) sub-quad connectivity of the snapshot points.")
      .def_property_readonly("snapshot_times", [](const solver::Solver& s) { return to_numpy(s.snapshot_times()); })
      .def("write_results", &solver::Solver::write_results, py::arg("path"));

  // ---- simulation
  py::class_<Discretization>(m, "Discretization")
      .def(py::init([](int order, int uniform_level, double f_max, double ppw, int max_level,
                       const std::vector<std::tuple<double, double, double, double, int>>& regions,
                       int azimuthal_order) {
             Discretization d;
             d.azimuthal_order = azimuthal_order;
             d.order = order;
             d.uniform_level = uniform_level;
             d.f_max = f_max;
             d.points_per_wavelength = ppw;
             d.max_level = max_level;
             for (const auto& [x0, x1, z0, z1, level] : regions) d.regions.push_back({x0, x1, z0, z1, level});
             return d;
           }),
           py::arg("order") = 4, py::arg("uniform_level") = 0, py::arg("f_max") = 0.0,
           py::arg("points_per_wavelength") = 5.0, py::arg("max_level") = 12,
           py::arg("regions") = std::vector<std::tuple<double, double, double, double, int>>{},
           py::arg("azimuthal_order") = 0,
           "Element order and refinement: uniform levels, then adaptive refinement to points_per_wavelength "
           "at f_max (0 = off) and to the levels of regions [(xmin, xmax, zmin, zmax, level), ...], "
           "capped at max_level; then 2:1 balance. azimuthal_order (axisymmetric meshes): 0 for "
           "axisymmetric fields (monopole), 1 for fields varying as cos(theta) (dipole).")
      .def_readonly("order", &Discretization::order)
      .def_readonly("uniform_level", &Discretization::uniform_level)
      .def_readonly("f_max", &Discretization::f_max)
      .def_readonly("points_per_wavelength", &Discretization::points_per_wavelength)
      .def_readonly("max_level", &Discretization::max_level)
      .def_readonly("azimuthal_order", &Discretization::azimuthal_order)
      .def_property_readonly("regions", [](const Discretization& d) {
        std::vector<std::tuple<double, double, double, double, int>> out;
        for (const auto& r : d.regions) out.emplace_back(r.xmin, r.xmax, r.zmin, r.zmax, r.level);
        return out;
      });

  py::class_<Simulation>(m, "Simulation")
      .def(py::init([](std::shared_ptr<mesh::MacroMesh> mesh, const Discretization& disc) {
             return Simulation(std::move(mesh), disc);
           }),
           py::arg("mesh"), py::arg("discretization") = Discretization{})
      .def("setup", &Simulation::setup, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("is_setup", &Simulation::is_setup)
      .def_property_readonly("mesh", [](const Simulation& s) {
        return std::const_pointer_cast<mesh::MacroMesh>(s.mesh_ptr());
      })
      .def_property_readonly("discretization", &Simulation::discretization)
      .def_property_readonly("forest", &Simulation::forest, py::return_value_policy::reference_internal)
      .def_property_readonly("dofs", &Simulation::dofs, py::return_value_policy::reference_internal)
      .def_property_readonly("geometry", &Simulation::geometry, py::return_value_policy::reference_internal)
      .def_property_readonly("gll", [](const Simulation& s) { return rule_to_tuple(s.gll()); })
      .def("geometry_kind_counts", &Simulation::geometry_kind_counts)
      .def("set_material", &Simulation::set_material, py::arg("region"), py::arg("material"))
      .def("set_boundary", [](Simulation& s, int tag, const std::string& type) {
        s.set_boundary(tag, boundary_type_from_string(type));
      }, py::arg("tag"), py::arg("type"), "type: 'free', 'rigid' or 'absorbing'.")
      .def("add_source", &Simulation::add_source, py::arg("source"))
      .def("clear_sources", &Simulation::clear_sources)
      .def("add_receiver", [](Simulation& s, const std::string& name, double x, double z,
                              std::vector<std::string> quantities, const std::string& medium) {
        s.add_receiver({name, {x, z}, std::move(quantities), medium});
      }, py::arg("name"), py::arg("x"), py::arg("z"), py::arg("quantities"), py::arg("medium") = "",
           "medium ('acoustic' / 'elastic') picks the side of a receiver on a fluid-solid interface.")
      .def("clear_receivers", &Simulation::clear_receivers)
      .def("set_time", [](Simulation& s, int n_steps, double duration, double dt, double safety,
                          int record_every, int energy_every, int snapshot_every,
                          std::vector<std::string> snapshot_quantities) {
        TimeConfig t;
        t.n_steps = n_steps;
        t.duration = duration;
        t.dt = dt;
        t.safety = safety;
        t.record_every = record_every;
        t.energy_every = energy_every;
        t.snapshot_every = snapshot_every;
        t.snapshot_quantities = std::move(snapshot_quantities);
        s.set_time(t);
      }, py::arg("n_steps") = 0, py::arg("duration") = 0.0, py::arg("dt") = 0.0, py::arg("safety") = 0.8,
           py::arg("record_every") = 1, py::arg("energy_every") = 0, py::arg("snapshot_every") = 0,
           py::arg("snapshot_quantities") = std::vector<std::string>{"velocity", "pressure"},
           "Time settings. dt = 0 picks safety * the estimated stability limit; n_steps = 0 derives it from duration.")
      .def_property_readonly("materials", &Simulation::materials)
      .def_property_readonly("boundaries", [](const Simulation& s) {
        std::map<int, std::string> out;
        for (const auto& [tag, type] : s.boundaries()) out[tag] = to_string(type);
        return out;
      })
      .def_property_readonly("sources", &Simulation::sources)
      .def_property_readonly("receivers", &Simulation::receivers)
      .def_property_readonly("time", &Simulation::time)
      .def_property_readonly("solver", &Simulation::solver, py::return_value_policy::reference_internal,
                             "Time-stepping state, built on first access.")
      .def("run", &Simulation::run, py::call_guard<py::gil_scoped_release>())
      .def("write_results", &Simulation::write_results, py::arg("path"))
      .def("save", &Simulation::save, py::arg("path"))
      .def_static("load", &Simulation::load, py::arg("path"));
}
