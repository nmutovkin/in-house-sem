"""Borehole (axisymmetric) monopole and dipole against the discrete-wavenumber reference."""

import numpy as np
import pytest

pytest.importorskip("scipy")

import insem  # noqa: E402
from insem import mesh, post  # noqa: E402
from insem.reference import borehole_axis_pressure, borehole_dipole_pressure, borehole_mode_slowness  # noqa: E402

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
    for order in (0, 1):
        assert np.abs(_reflection([(MUD, 0.1)], MUD, w, k, order)).max() < 1e-12
        assert np.abs(_reflection([(MUD, 0.1), (MUD, 0.13)], MUD, w, k, order)).max() < 1e-12


def test_dipole_reference_in_free_space():
    import analytic

    t = np.arange(3000) * 2e-6
    stf = insem.ricker(8000.0)(t)
    p = borehole_dipole_pressure(t, stf, [0.5, 1.0], 0.05, [(MUD, 0.1)], MUD)
    for j, z in enumerate([0.5, 1.0]):
        exact = analytic.acoustic_dipole_pressure_3d(t, stf, 0.05, z, MUD.vp)
        assert analytic.rel_l2(p[:, j], exact) < 1e-4  # 3e-5
    with pytest.raises(ValueError):
        borehole_dipole_pressure(t, stf, [0.5], 0.0, [(MUD, 0.1)], MUD)  # P = 0 on the axis


SLOW_FORMATION = insem.Material.elastic(2500.0, 1000.0, 2200.0)  # vs < mud vp: no refracted shear


@pytest.fixture(scope="module")
def dipole_open_hole():
    # Dipole (azimuthal order 1) in a slow formation, receivers at half the
    # borehole radius. The outer boundary is far enough that the formation P
    # wave it reflects (a few % with first-order Stacey) arrives after the
    # recording window.
    f0, offsets = 3000.0, np.round(np.arange(1.0, 2.01, 0.2), 3)
    b = mesh.borehole((-1.5, 3.5), [("mud", 0.1)], 4.0, dz=0.08, dr=0.3)
    sim = insem.Simulation(b.mesh, insem.Discretization(order=4, f_max=2.5 * f0, azimuthal_order=1))
    sim.set_material(b.regions["mud"], MUD)
    sim.set_material(b.regions["formation"], SLOW_FORMATION)
    for name in ("bottom", "outer", "top"):
        sim.set_boundary(b.boundaries[name], "absorbing")
    stf = insem.ricker(f0)
    sim.add_source(insem.Source.dipole(0.0, 0.0, 1.0, 0.0, stf))
    for z in offsets:
        sim.add_receiver(f"z{z}", 0.05, float(z), ["pressure"])
    sim.set_time(duration=3.2e-3, record_every=4)
    sim.run()
    s = sim.solver
    t = s.times
    sem = np.array([s.trace(f"z{z}", "pressure")[:, 0] for z in offsets])
    ref = borehole_dipole_pressure(t, stf(t), offsets, 0.05, [(MUD, 0.1)], SLOW_FORMATION).T
    return t, offsets, sem, ref


@pytest.mark.slow
def test_dipole_open_hole_waveforms(dipole_open_hole):
    _, _, sem, ref = dipole_open_hole
    errors = [np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(sem, ref)]
    assert max(errors) < 0.015  # measured 0.3-0.4 %


@pytest.mark.slow
def test_dipole_open_hole_stc(dipole_open_hole):
    # The flexural wave: 1060 us/m at 3 kHz in both, slower than the shear
    # slowness 1/vs = 1000 us/m it tends to at low frequency (dispersion).
    t, offsets, sem, ref = dipole_open_hole
    grid = np.linspace(600e-6, 1600e-6, 1001)
    times = np.arange(0, 2.0e-3, 5e-6)
    picks = [post.stc(d, offsets, t[1] - t[0], grid, window=0.5e-3, times=times).peak()[0] for d in (sem, ref)]
    assert picks[0] == pytest.approx(picks[1], abs=3e-6)
    assert 1 / SLOW_FORMATION.vs < picks[0] < 1.15 / SLOW_FORMATION.vs


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


@pytest.mark.slow
def test_dipole_flexural_dispersion(dipole_open_hole):
    # Phase slowness of the flexural wave from the arrays (frequency-domain
    # semblance) against modal root-finding of the dispersion relation. The
    # SEM and reference arrays agree closely; with a 1 m aperture both sit
    # within 0.6 % of the modal curve at 3-5 kHz (1.2 % at 2.5 kHz, where
    # the source has less energy and the aperture is shorter in wavelengths).
    t, offsets, sem, ref = dipole_open_hole
    freqs = np.arange(3000.0, 5001.0, 500.0)
    grid = np.linspace(800e-6, 1400e-6, 1201)
    picks = [post.dispersion(d, offsets, t[1] - t[0], freqs, grid, mute=(1.2e-3, 0.95e-3)).peak()[0]
             for d in (sem, ref)]
    np.testing.assert_allclose(picks[0], picks[1], atol=2e-6)
    modal = borehole_mode_slowness(freqs, [(MUD, 0.1)], SLOW_FORMATION, order=1)
    np.testing.assert_allclose(picks[0], modal, rtol=0.01)


def test_mode_slowness_limits():
    # Stoneley (order 0, fast formation): the tube-wave slowness at low
    # frequency, and the 704 us/m that STC picks from the 5 kHz arrays above.
    mu = FORMATION.rho * FORMATION.vs**2
    tube = np.sqrt(1 + MUD.rho * MUD.vp**2 / mu) / MUD.vp
    st = borehole_mode_slowness([300.0, 5000.0], [(MUD, 0.1)], FORMATION, order=0)
    assert st[0] == pytest.approx(tube, rel=1e-3)
    assert st[1] == pytest.approx(704e-6, abs=2e-6)
    # Flexural (order 1, slow formation): tends to the shear slowness at low
    # frequency and slows down with frequency.
    fl = borehole_mode_slowness([1000.0, 2000.0, 4000.0], [(MUD, 0.1)], SLOW_FORMATION, order=1)
    assert fl[0] == pytest.approx(1 / SLOW_FORMATION.vs, rel=1e-4)
    assert np.all(np.diff(fl) > 0) and fl[-1] < 1.1 / SLOW_FORMATION.vs
