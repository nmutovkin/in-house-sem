"""Azimuthal order 1 (dipole) solver against 3D analytic solutions (docs/DESIGN.md §2.5)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

import insem  # noqa: E402
from insem import mesh  # noqa: E402

import analytic  # noqa: E402


def dipole_box(material, order, nr=24, nz=48, r1=4.0, half=4.0):
    m = mesh.rectangle((0, r1), (-half, half), nx=nr, nz=nz, coords="axisymmetric")
    sim = insem.Simulation(m, insem.Discretization(order=order, azimuthal_order=1))
    sim.set_material(0, material)
    for tag in (mesh.RIGHT, mesh.TOP, mesh.BOTTOM):  # LEFT is the axis
        sim.set_boundary(tag, "absorbing")
    return sim


def test_fluid_dipole_on_axis():
    c, rho = 1.0, 1.3
    receivers = {"near": (0.3, 0.4), "off": (1.2, -0.9), "up": (0.8, 1.6)}
    errors = {}
    for order in (4, 6):
        sim = dipole_box(insem.Material.acoustic(c, rho), order)
        stf = insem.ricker(1.0)
        sim.add_source(insem.Source.dipole(0.0, 0.0, 1.0, 0.0, stf))
        for name, (r, z) in receivers.items():
            sim.add_receiver(name, r, z, ["pressure"])
        sim.set_time(duration=4.5)
        sim.run()
        s = sim.solver
        for name, (r, z) in receivers.items():
            exact = analytic.acoustic_dipole_pressure_3d(s.times, stf(s.times), r, z, c)
            errors[(order, name)] = analytic.rel_l2(s.trace(name, "pressure")[:, 0], exact)
    print(errors)
    for name in receivers:
        assert errors[(4, name)] < 0.015
        assert errors[(6, name)] < 0.5 * errors[(4, name)]


def test_elastic_horizontal_force_on_axis():
    vp, vs, rho = 2.0, 1.15, 1.0
    sim = dipole_box(insem.Material.elastic(vp, vs, rho), 5, r1=5.0, half=5.0)
    stf = insem.ricker(1.0)
    sim.add_source(insem.Source.force(0.0, 0.0, 1.0, 0.0, stf))
    receivers = {"axis": (0.0, 1.8), "off": (1.5, 1.0), "off2": (2.0, -1.2)}
    for name, (r, z) in receivers.items():
        sim.add_receiver(name, r, z, ["displacement"])
    sim.set_time(duration=4.0)
    sim.run()
    s = sim.solver
    for name, (r, z) in receivers.items():
        u = s.trace(name, "displacement")
        assert u.shape[1] == 3  # (U_r, U_theta, U_z)
        exact = analytic.elastic_horizontal_force_3d(s.times, stf(s.times), r, z, vp, vs, rho)
        err = analytic.rel_l2(u, exact)
        print(name, err)
        assert err < 0.012


def test_off_axis_ring_tends_to_the_point_force():
    # A cos(theta) ring of net force fx close to the axis radiates like the
    # point force on the axis.
    vp, vs, rho = 2.0, 1.15, 1.0
    traces = []
    for rs in (0.0, 0.02):
        sim = dipole_box(insem.Material.elastic(vp, vs, rho), 4, nr=12, nz=24, r1=4.0)
        sim.add_source(insem.Source.force(rs, 0.0, 1.0, 0.0, insem.ricker(1.0)))
        sim.add_receiver("r", 1.5, 1.0, ["displacement"])
        sim.set_time(duration=3.0)
        sim.run()
        traces.append(sim.solver.trace("r", "displacement"))
    assert analytic.rel_l2(traces[1], traces[0]) < 0.01


@pytest.fixture(scope="module")
def dipole_run(tmp_path_factory):
    """Small coupled dipole run (fluid column in a solid) with snapshots."""
    m = mesh.rectangle((0, 1.0), (-1.0, 1.0), nx=4, nz=8, coords="axisymmetric")
    material = np.where(m.nodes[m.cells].mean(axis=1)[:, 0] < 0.25, 0, 1)
    m = insem.MacroMesh(m.cell_type, m.coords, m.nodes, m.cells, material, m.boundary)
    sim = insem.Simulation(m, insem.Discretization(order=3, azimuthal_order=1))
    sim.set_material(0, insem.Material.acoustic(1.5, 1.0))
    sim.set_material(1, insem.Material.elastic(3.0, 1.2, 2.0))
    for tag in (mesh.BOTTOM, mesh.RIGHT, mesh.TOP):
        sim.set_boundary(tag, "absorbing")
    sim.add_source(insem.Source.dipole(0.0, 0.0, 1.0, 0.0, insem.ricker(2.0)))
    sim.add_receiver("h", 0.125, 0.5, ["pressure"])
    sim.add_receiver("g", 0.5, 0.5, ["displacement"])  # a GLL point (element corner)
    sim.set_time(duration=1.5, snapshot_every=10, snapshot_quantities=["displacement", "pressure"])
    sim.run()
    path = tmp_path_factory.mktemp("dipole") / "results.h5"
    sim.write_results(str(path))
    return sim, path


def test_dipole_results_and_exports(dipole_run, tmp_path):
    sim, path = dipole_run
    s = sim.solver
    res = insem.post.read_results(path)
    assert res.coords == "axisymmetric" and res.azimuthal_order == 1
    g = res.trace("g", "displacement")
    assert g.shape[1] == 3 and np.abs(g).max(axis=0).min() > 0  # U_r, U_theta, U_z
    # Traces and snapshots are (U_r, U_theta, U_z); the state is (U+, U-, U_z).
    u, _, _ = s.state("elastic")
    pts = s.coordinates("elastic")
    i = np.argmin((pts[0] - 0.5) ** 2 + (pts[1] - 0.5) ** 2)
    ur, ut = (u[0, i] + u[1, i]) / np.sqrt(2), (u[0, i] - u[1, i]) / np.sqrt(2)
    np.testing.assert_allclose(g[-1], [ur, ut, u[2, i]], rtol=1e-5, atol=1e-6 * np.abs(g).max())
    snap = s.snapshot("elastic", "displacement")
    spts = s.snapshot_points("elastic")
    j = np.argmin((spts[0] - 0.5) ** 2 + (spts[1] - 0.5) ** 2)
    np.testing.assert_allclose(snap[-1, :, j], g[np.argmin(np.abs(s.times - s.snapshot_times[-1]))],
                               rtol=1e-4, atol=1e-5 * np.abs(g).max())
    # Component names in the exports.
    out = insem.post.write_segy(path, tmp_path / "theta.sgy", "displacement", component="theta")
    np.testing.assert_allclose(insem.post.read_segy(out).data[0], g[:, 1], rtol=1e-6, atol=1e-7 * np.abs(g).max())
    with pytest.raises(ValueError, match="no component 'x'"):
        insem.post.write_segy(path, tmp_path / "x.sgy", "displacement", component="x")
    xdmf = insem.post.write_xdmf(path, tmp_path / "snap.xdmf").read_text()
    assert "displacement_theta" in xdmf and "displacement_r" in xdmf
