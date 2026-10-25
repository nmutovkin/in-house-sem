"""Borehole (axisymmetric) monopole against the discrete-wavenumber reference."""

import numpy as np
import pytest

pytest.importorskip("scipy")

import insem  # noqa: E402
from insem import mesh, post  # noqa: E402
from insem.reference import borehole_axis_pressure  # noqa: E402

MUD = insem.Material.acoustic(1500.0, 1000.0)
FORMATION = insem.Material.elastic(4000.0, 2300.0, 2500.0)  # fast formation
F0 = 5000.0
OFFSETS = np.round(np.arange(1.0, 2.41, 0.2), 3)


def borehole_run(annuli, materials, offsets, duration, record_every=2):
    b = mesh.borehole((-1.0, 3.5), annuli, 3.0, dz=0.08, dr=0.3)
    sim = insem.Simulation(b.mesh, insem.Discretization(order=4, f_max=2.5 * F0))
    for name, mat in materials.items():
        sim.set_material(b.regions[name], mat)
    for name in ("bottom", "outer", "top"):
        sim.set_boundary(b.boundaries[name], "absorbing")
    stf = insem.ricker(F0)
    sim.add_source(insem.Source.monopole(0.0, 0.0, stf))
    for z in offsets:
        sim.add_receiver(f"z{z}", 0.0, float(z), ["pressure"])
    sim.set_time(duration=duration, record_every=record_every)
    sim.run()
    s = sim.solver
    return s.times, np.array([s.trace(f"z{z}", "pressure")[:, 0] for z in offsets]), stf


@pytest.fixture(scope="module")
def open_hole():
    t, sem, stf = borehole_run([("mud", 0.1)], {"mud": MUD, "formation": FORMATION}, OFFSETS, 2.0e-3)
    ref = borehole_axis_pressure(t, stf(t), OFFSETS, [(MUD, 0.1)], FORMATION).T
    return t, sem, ref


def test_open_hole_waveforms(open_hole):
    _, sem, ref = open_hole
    errors = [np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(sem, ref)]
    assert max(errors) < 0.02  # measured 0.2-0.9 %


def test_open_hole_stc(open_hole):
    # STC picks of the SEM array equal those of the reference: P (258 us/m;
    # at these short offsets the early arrival is 3 % slower than 1/vp in the
    # exact solution too), S (440 vs 1/vs = 435) and Stoneley (704 us/m,
    # close to the low-frequency tube-wave slowness of 721 us/m).
    t, sem, ref = open_hole
    grid = np.linspace(150e-6, 900e-6, 751)
    times = np.arange(0, 1.6e-3, 5e-6)
    picks = []
    for data in (sem, ref):
        r = post.stc(data, OFFSETS, t[1] - t[0], grid, window=0.2e-3, times=times)
        picks.append([r.peak(s_max=350e-6)[0], r.peak(s_min=380e-6, s_max=520e-6)[0], r.peak(s_min=550e-6)[0]])
    np.testing.assert_allclose(picks[0], picks[1], atol=2.5e-6)
    p, s, st = picks[0]
    assert p == pytest.approx(1 / FORMATION.vp, rel=0.05)
    assert s == pytest.approx(1 / FORMATION.vs, rel=0.03)
    mu = FORMATION.rho * FORMATION.vs**2
    tube = np.sqrt(1 + MUD.rho * MUD.vp**2 / mu) / MUD.vp
    assert st == pytest.approx(tube, rel=0.04)


@pytest.mark.slow
def test_cased_hole_waveforms():
    # Steel casing (9 mm) and cement: thin annuli make this run slower.
    steel = insem.Material.elastic(5900.0, 3200.0, 7850.0)
    cement = insem.Material.elastic(3000.0, 1700.0, 1900.0)
    annuli = [("mud", 0.101), ("steel", 0.110), ("cement", 0.150)]
    offsets = [1.0, 2.0]
    t, sem, stf = borehole_run(annuli, {"mud": MUD, "steel": steel, "cement": cement, "formation": FORMATION},
                               offsets, 2.0e-3, record_every=4)
    ref = borehole_axis_pressure(t, stf(t), offsets, [(MUD, 0.101), (steel, 0.110), (cement, 0.150)], FORMATION).T
    errors = [np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(sem, ref)]
    assert max(errors) < 0.02  # measured 0.6 % and 0.2 %


def test_reference_without_contrast_reflects_nothing():
    from insem.reference.borehole import _reflection

    k = np.linspace(0, 200, 11)
    w = 2 * np.pi * 5000 + 400j
    assert np.abs(_reflection([(MUD, 0.1)], MUD, w, k)).max() < 1e-12
    assert np.abs(_reflection([(MUD, 0.1), (MUD, 0.13)], MUD, w, k)).max() < 1e-12


def test_stc_recovers_plane_wave_slowness():
    dt, s_true = 1e-5, 312e-6
    t = np.arange(400) * dt
    offsets = np.arange(6) * 0.15
    x = np.array([np.exp(-((t - 1e-3 - s_true * z) / 1e-4) ** 2) for z in offsets])
    r = post.stc(x, offsets, dt, np.linspace(100e-6, 600e-6, 501), window=4e-4)
    s, T, c = r.peak()
    assert s == pytest.approx(s_true, abs=1.5e-6)
    assert c > 0.99


def test_borehole_template():
    b = mesh.borehole((-1.0, 3.0), [("mud", 0.1), ("casing", 0.11), ("cement", 0.15)], 2.0,
                      beds=[(-1.0, "shale"), (0.5, "sand"), (1.7, "shale")], dz=0.05)
    assert b.regions == {"mud": 1, "casing": 2, "cement": 3, "shale": 4, "sand": 5}
    for v in (0.1, 0.11, 0.15, 2.0):
        assert np.isclose(b.r, v).any()
    for v in (-1.0, 0.5, 1.7, 3.0):
        assert np.isclose(b.z, v).any()
    assert np.all(np.diff(b.r) > 0) and np.diff(b.r).max() <= 4 * 0.05 * 1.5 + 1e-12
    sim = insem.Simulation(b.mesh, insem.Discretization(order=2))
    sim.setup()
    m = b.mesh
    cells = m.cells
    centre = m.nodes[cells].mean(axis=1)
    sand = m.material == b.regions["sand"]
    assert np.all(centre[sand, 0] > 0.15) and np.all((centre[sand, 1] > 0.5) & (centre[sand, 1] < 1.7))
    with pytest.raises(ValueError):
        mesh.borehole((0, 1), [("mud", 0.2), ("casing", 0.1)], 1.0)
