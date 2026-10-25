// Python bindings: the C++ object model exposed as insem._core. The pure
// Python layer in python/insem builds on these.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
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
#include "insem/sim/simulation.hpp"

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
      .def("leaf_box", &mesh::Forest::leaf_box, py::arg("leaf"));

  // ---- ops
  py::class_<ops::DofMap>(m, "DofMap")
      .def_static("build", &ops::DofMap::build, py::arg("forest"), py::arg("ngll"))
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

  // ---- simulation
  py::class_<Discretization>(m, "Discretization")
      .def(py::init([](int order, int uniform_level) { return Discretization{order, uniform_level}; }),
           py::arg("order") = 4, py::arg("uniform_level") = 0)
      .def_readwrite("order", &Discretization::order)
      .def_readwrite("uniform_level", &Discretization::uniform_level);

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
      .def("save", &Simulation::save, py::arg("path"))
      .def_static("load", &Simulation::load, py::arg("path"));
}
