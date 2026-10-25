from pathlib import Path

import numpy as np
import pytest

import insem
from insem import mesh

DATA = Path(__file__).parent / "data"


def area(m, order=4):
    sim = insem.Simulation(m, insem.Discretization(order=order))
    sim.setup()
    _, w = sim.gll
    return float(np.einsum("i,j,eji->", w, w, sim.geometry.jac)), sim


@pytest.mark.parametrize("name", ["layered_q4_v41.msh", "layered_q4_v22.msh", "layered_q9_v41.msh"])
def test_layered_gmsh_files(name):
    g = mesh.read_gmsh(DATA / name)
    m = g.mesh
    assert m.cell_type == ("quad9" if "q9" in name else "quad4")
    assert set(m.material) == {1, 2}
    assert set(m.boundary[:, 2]) == {11, 12, 13}
    assert g.regions == {"rock": 1, "water": 2}
    assert g.boundaries == {"bottom": 11, "top": 12, "sides": 13}
    a, sim = area(m)
    assert a == pytest.approx(4.0 * 3.5, rel=1e-12)
    # Every material region is where the geometry says it is.
    xz = sim.geometry.x.mean(axis=(1, 2)), sim.geometry.z.mean(axis=(1, 2))
    leaves = sim.forest.leaves[:, 0]
    region = m.material[leaves]
    assert np.all(xz[1][region == 1] < 2.0) and np.all(xz[1][region == 2] > 2.0)


def test_v22_and_v41_files_hold_the_same_mesh():
    a = mesh.read_gmsh(DATA / "layered_q4_v41.msh").mesh
    b = mesh.read_gmsh(DATA / "layered_q4_v22.msh").mesh
    assert a.n_cells == b.n_cells
    np.testing.assert_allclose(np.sort(a.nodes, axis=0), np.sort(b.nodes, axis=0))


def test_curved_annulus():
    g = mesh.read_gmsh(DATA / "annulus_q9_v41.msh")
    assert g.mesh.cell_type == "quad9"
    a, sim = area(g.mesh, order=5)
    assert a == pytest.approx(0.25 * np.pi * (4 - 1), rel=1e-4)
    assert sim.geometry_kind_counts()["curved"] > 0


def test_coupled_run_on_gmsh_mesh():
    g = mesh.read_gmsh(DATA / "layered_q4_v41.msh")
    sim = insem.Simulation(g.mesh, insem.Discretization(order=4))
    sim.set_material(g.regions["rock"], insem.Material.elastic(3.0, 1.7, 2.3))
    sim.set_material(g.regions["water"], insem.Material.acoustic(1.5, 1.0))
    for name in ("bottom", "sides"):
        sim.set_boundary(g.boundaries[name], "absorbing")
    sim.set_boundary(g.boundaries["top"], "free")
    sim.add_source(insem.Source.monopole(2.0, 2.8, insem.ricker(2.0)))
    sim.add_receiver("geo", 2.5, 1.0, ["velocity"])
    sim.set_time(duration=2.0)
    sim.run()
    v = sim.solver.trace("geo", "velocity")
    assert np.all(np.isfinite(v)) and np.abs(v).max() > 0


def _write(tmp_path, body):
    p = tmp_path / "m.msh"
    p.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n" + body)
    return p


def test_clockwise_cells_are_reoriented(tmp_path):
    p = _write(tmp_path, "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
                         "$Elements\n2\n1 3 2 7 1 1 4 3 2\n2 1 2 9 1 1 2\n$EndElements\n")
    g = mesh.read_gmsh(p)
    a, _ = area(g.mesh)
    assert a == pytest.approx(1.0)
    assert list(g.mesh.material) == [7]
    assert g.mesh.boundary.tolist() == [[0, 0, 9]] or g.mesh.boundary[0, 2] == 9


def test_triangles_are_rejected(tmp_path):
    p = _write(tmp_path, "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
                         "$Elements\n1\n1 2 2 1 1 1 2 3\n$EndElements\n")
    with pytest.raises(ValueError, match="SubdivisionAlgorithm"):
        mesh.read_gmsh(p)


def test_plane_selection(tmp_path):
    p = _write(tmp_path, "$Nodes\n4\n1 0 5 0\n2 2 5 0\n3 2 5 1\n4 0 5 1\n$EndNodes\n"
                         "$Elements\n1\n1 3 2 1 1 1 2 3 4\n$EndElements\n")
    g = mesh.read_gmsh(p, plane="xz")
    a, _ = area(g.mesh)
    assert a == pytest.approx(2.0)
