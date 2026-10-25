import numpy as np
import pytest

import insem
from insem import basis, mesh


def test_version_and_precision():
    assert insem.__version__ == "0.1.0"
    assert insem.real_dtype in ("float32", "float64")


def test_gll_matches_numpy_legendre():
    for n in range(2, 12):
        x, w = basis.gll(n)
        # Interior GLL points are the roots of P'_{n-1}.
        dp = np.polynomial.legendre.Legendre.basis(n - 1).deriv()
        np.testing.assert_allclose(x[1:-1], np.sort(dp.roots()), atol=1e-13)
        assert w.sum() == pytest.approx(2.0, abs=1e-13)


def test_glj01_matches_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    for n in range(3, 10):
        x, w = basis.glj01(n)
        # Interior points: roots of P_{n-2}^{(1,2)}.
        roots, _ = scipy_special.roots_jacobi(n - 2, 1.0, 2.0)
        np.testing.assert_allclose(x[1:-1], np.sort(roots), atol=1e-13)
        assert w.sum() == pytest.approx(2.0, abs=1e-13)  # int (1 + x) dx


def test_derivative_matrix_differentiates_polynomials():
    x, _ = basis.gll(7)
    d = basis.derivative_matrix(x)
    f = x**6 - 2 * x**3 + 1
    np.testing.assert_allclose(d @ f, 6 * x**5 - 6 * x**2, atol=1e-11)


def test_rectangle_template_and_simulation():
    m = mesh.rectangle((0.0, 2.0), (0.0, 1.0), nx=4, nz=2)
    assert m.n_cells == 8
    assert m.cell_type == "quad4"
    assert set(m.boundary[:, 2]) == {mesh.BOTTOM, mesh.RIGHT, mesh.TOP, mesh.LEFT}
    sim = insem.Simulation(m, insem.Discretization(order=4, uniform_level=1))
    sim.setup()
    assert sim.dofs.n_elements == 32
    assert sim.dofs.n_global == (4 * 2 * 4 + 1) * (2 * 2 * 4 + 1)
    assert sim.geometry_kind_counts()["rect"] == 32


def test_explicit_breakpoints_and_material_function():
    m = mesh.rectangle([0.0, 0.1, 0.3, 1.0], [-2.0, -1.0, 0.0],
                       material=lambda x, z: 1 if x < 0.1 else 2, coords="axisymmetric")
    assert m.coords == "axisymmetric"
    assert list(m.material) == [1, 2, 2, 1, 2, 2]


def test_quadratic_rectangle_area():
    m = mesh.rectangle((0.0, 3.0), (1.0, 2.0), nx=3, nz=2, quadratic=True)
    sim = insem.Simulation(m, insem.Discretization(order=5))
    sim.setup()
    _, w = sim.gll
    area = np.einsum("i,j,eji->", w, w, sim.geometry.jac)
    assert area == pytest.approx(3.0, rel=1e-13)


def test_gather_assemble_transpose():
    m = mesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=3, nz=3)
    sim = insem.Simulation(m, insem.Discretization(order=3, uniform_level=1))
    sim.setup()
    d = sim.dofs
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, d.n_global))
    loc = rng.standard_normal((d.n_elements, 2, d.ngll, d.ngll))
    lhs = np.sum(d.assemble(loc) * g)
    rhs = np.sum(loc * d.gather(g))
    assert lhs == pytest.approx(rhs, rel=1e-12)
    np.testing.assert_array_equal(d.assemble(d.gather(np.ones(d.n_global))), d.multiplicity)


def test_global_coordinates_consistent():
    m = mesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=2, nz=2, quadratic=True)
    sim = insem.Simulation(m, insem.Discretization(order=4, uniform_level=2))
    sim.setup()
    xz, mismatch = insem.global_coordinates(sim.dofs, sim.geometry)
    assert xz.shape == (2, sim.dofs.n_global)
    assert mismatch < 1e-14
    assert xz.min() == pytest.approx(0.0) and xz.max() == pytest.approx(1.0)


def test_case_file_round_trip(tmp_path):
    h5py = pytest.importorskip("h5py")
    m = mesh.rectangle((0.0, 1.0), (0.0, 2.0), nx=2, nz=3, material=[0, 1, 0, 1, 0, 1])
    sim = insem.Simulation(m, insem.Discretization(order=6, uniform_level=1))
    path = tmp_path / "case.h5"
    sim.save(str(path))
    with h5py.File(path, "r") as f:
        assert f.attrs["format"] == b"insem-case" or f.attrs["format"] == "insem-case"
        assert f["mesh/cells"].shape == (6, 4)
    loaded = insem.Simulation.load(str(path))
    assert loaded.discretization.order == 6
    np.testing.assert_array_equal(loaded.mesh.material, m.material)
    np.testing.assert_array_equal(loaded.mesh.boundary, m.boundary)


def test_errors_are_reported():
    with pytest.raises(insem.InsemError, match="counter-clockwise"):
        insem.MacroMesh("quad4", "cartesian", np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
                        np.array([[0, 3, 2, 1]]), np.array([0]))
    with pytest.raises(ValueError):
        mesh.rectangle((1.0, 0.0), (0.0, 1.0), nx=2, nz=2)
