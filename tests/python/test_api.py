import numpy as np
import pytest

import insem
from insem import mesh


def test_source_time_functions():
    r = insem.ricker(10.0)
    assert r.t0 == pytest.approx(0.12)
    assert r(0.12) == pytest.approx(1.0)
    t = np.linspace(0, 0.3, 301)
    np.testing.assert_allclose(r(t), [r(x) for x in t])
    g = insem.gaussian(10.0, t0=0.1, amplitude=2.0)
    assert g(0.1) == pytest.approx(2.0)
    dg = insem.gaussian_derivative(10.0, t0=0.1)
    h = 1e-6
    assert dg(0.09) == pytest.approx((insem.gaussian(10.0, 0.1)(0.09 + h) - insem.gaussian(10.0, 0.1)(0.09 - h)) / (2 * h), rel=1e-5)
    s = insem.sampled_stf([0.0, 1.0, 0.0], dt=0.5)
    assert s(0.25) == pytest.approx(0.5)
    assert s(2.0) == 0.0


def test_material_validation():
    with pytest.raises(insem.InsemError):
        insem.Material.elastic(1.0, 1.5, 1.0)  # vs > vp
    with pytest.raises(insem.InsemError):
        insem.Material.acoustic(1500.0, -1.0)
    with pytest.raises(insem.InsemError, match="acoustic material"):
        insem.Material.elastic(1.0, 0.0, 1.0)


def test_layered_template():
    m = mesh.layered((0, 4), [-3.0, -1.0, lambda x: 0.1 * x, 2.0], nz=[2, 3, 1], regions=[7, 8, 9], nx=4)
    assert m.n_cells == 4 * 6
    assert list(m.material) == [7] * 8 + [8] * 12 + [9] * 4
    nodes = m.nodes
    # The curved interface row lies exactly on z = 0.1 x.
    row = nodes[(2 + 3) * 5:(2 + 3) * 5 + 5]
    np.testing.assert_allclose(row[:, 1], 0.1 * row[:, 0])
    with pytest.raises(ValueError):
        mesh.layered((0, 1), [0.0, 1.0, 0.5], nz=1, regions=[0, 1], nx=1)


def test_case_file_round_trips_the_full_description(tmp_path):
    m = mesh.rectangle((0, 2), (0, 1), nx=4, nz=2)
    sim = insem.Simulation(m, insem.Discretization(order=3))
    sim.set_material(0, insem.Material.elastic(3.0, 1.7, 2.2))
    sim.set_boundary(mesh.TOP, "free")
    sim.set_boundary(mesh.BOTTOM, "absorbing")
    sim.add_source(insem.Source.moment_tensor(1.0, 0.5, 1.0, -1.0, 0.25, insem.ricker(5.0, t0=0.3, amplitude=2.0)))
    sim.add_source(insem.Source.force(0.5, 0.5, 1.0, 0.0, insem.sampled_stf([0.0, 1.0, 0.5], 0.01)))
    sim.add_receiver("geo1", 1.5, 0.2, ["velocity", "displacement"])
    sim.set_time(n_steps=40, record_every=2, energy_every=5)
    path = tmp_path / "case.h5"
    sim.save(str(path))
    other = insem.Simulation.load(str(path))
    assert other.materials[0].vs == pytest.approx(1.7)
    assert other.boundaries == {mesh.TOP: "free", mesh.BOTTOM: "absorbing"}
    assert [s.kind for s in other.sources] == ["moment_tensor", "force"]
    assert other.sources[0].stf.amplitude == 2.0
    assert other.receivers[0].quantities == ["velocity", "displacement"]
    assert other.time.record_every == 2
    sim.run()
    other.run()
    np.testing.assert_array_equal(sim.solver.trace("geo1", "velocity"), other.solver.trace("geo1", "velocity"))


def test_results_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    m = mesh.rectangle((0, 1), (0, 1), nx=4, nz=4)
    sim = insem.Simulation(m, insem.Discretization(order=4))
    sim.set_material(0, insem.Material.acoustic(1.5, 1.0))
    sim.add_source(insem.Source.monopole(0.5, 0.5, insem.ricker(5.0)))
    sim.add_receiver("h1", 0.2, 0.3, ["pressure", "particle_velocity"])
    sim.set_time(n_steps=30, energy_every=10)
    sim.run()
    path = tmp_path / "out.h5"
    sim.write_results(str(path))
    with h5py.File(path, "r") as f:
        assert f["receivers/h1/pressure"].shape == (31, 1)
        assert f["receivers/h1/particle_velocity"].shape == (31, 2)
        assert f["energy"].shape == (4, 3)
        np.testing.assert_allclose(f["receivers/h1/pressure"][:], sim.solver.trace("h1", "pressure"))
        assert f["receivers/h1"].attrs["x"] == pytest.approx(0.2)


def test_solver_state_and_reset():
    m = mesh.rectangle((0, 1), (0, 1), nx=2, nz=2)
    sim = insem.Simulation(m, insem.Discretization(order=3))
    sim.set_material(0, insem.Material.elastic(2.0, 1.0, 1.0))
    sim.set_time(n_steps=5)
    s = sim.solver
    assert s.media == ["elastic"]
    u, v, a = s.state("elastic")
    assert u.shape == (2, sim.dofs.n_global)
    s.advance(2)
    assert s.step == 2
    with pytest.raises(insem.InsemError):
        s.set_state("elastic", u, v)  # only before the first step
    sim.set_time(n_steps=10)  # changing the description rebuilds the solver
    assert sim.solver.step == 0
    assert sim.solver.n_steps == 10


def test_adaptive_discretization_round_trip(tmp_path):
    m = mesh.layered((0, 4), [-2.0, 0.0, 1.0], nz=[2, 1], regions=[0, 1], nx=4)
    disc = insem.Discretization(order=3, f_max=2.0, points_per_wavelength=6.0, max_level=4,
                                regions=[(1.0, 2.0, -0.5, 0.5, 2)])
    sim = insem.Simulation(m, disc)
    sim.set_material(0, insem.Material.elastic(4.0, 2.0, 2.0))
    sim.set_material(1, insem.Material.acoustic(1.0, 1.0))
    sim.setup()
    levels = sim.forest.leaves[:, 1]
    assert levels.max() >= 2
    # 2:1 balance across all faces.
    for leaf in range(sim.forest.n_leaves):
        for face in range(4):
            nb, _, diff = sim.forest.face_neighbor(leaf, face)
            if nb >= 0:
                assert abs(diff) <= 1
    path = tmp_path / "case.h5"
    sim.save(str(path))
    other = insem.Simulation.load(str(path))
    assert other.discretization.f_max == 2.0
    assert other.discretization.regions == [(1.0, 2.0, -0.5, 0.5, 2)]
    other.setup()
    np.testing.assert_array_equal(other.forest.leaves, sim.forest.leaves)
