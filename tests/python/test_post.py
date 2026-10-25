import xml.etree.ElementTree as ET

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import insem  # noqa: E402
from insem import mesh, post  # noqa: E402


@pytest.fixture(scope="module")
def coupled_run(tmp_path_factory):
    """Water over rock with a refined box, snapshots and a receiver line."""
    tmp = tmp_path_factory.mktemp("run")
    m = mesh.layered((0, 4), [-2.0, 0.0, 1.5], nz=[4, 3], regions=[1, 2], nx=8)
    sim = insem.Simulation(m, insem.Discretization(order=3, regions=[(1.5, 2.5, -0.5, 0.5, 1)]))
    sim.set_material(1, insem.Material.elastic(3.0, 1.7, 2.3))
    sim.set_material(2, insem.Material.acoustic(1.5, 1.0))
    for tag in (mesh.BOTTOM, mesh.LEFT, mesh.RIGHT):
        sim.set_boundary(tag, "absorbing")
    sim.add_source(insem.Source.monopole(2.0, 0.8, insem.ricker(2.0)))
    for i, x in enumerate(np.linspace(0.5, 3.5, 4)):
        sim.add_receiver(f"g{i}", float(x), -0.5, ["velocity", "displacement"])
    sim.add_receiver("h", 1.0, 1.0, ["pressure"])
    sim.set_time(duration=1.5, record_every=2, snapshot_every=20, energy_every=10)
    sim.run()
    path = tmp / "results.h5"
    sim.write_results(str(path))
    return sim, path


def test_read_results(coupled_run):
    sim, path = coupled_run
    res = post.read_results(path)
    s = sim.solver
    np.testing.assert_array_equal(res.time, s.times)
    np.testing.assert_array_equal(res.trace("g2", "velocity"), s.trace("g2", "velocity"))
    assert res.receivers["h"].medium == "acoustic"
    assert res.receivers["g1"].z == pytest.approx(-0.5)
    assert res.energy.shape[1] == 3
    assert set(res.snapshots) == {("elastic", "velocity"), ("acoustic", "pressure")}
    np.testing.assert_array_equal(res.snapshots[("elastic", "velocity")], s.snapshot("elastic", "velocity"))
    points, cells = res.snapshot_mesh["acoustic"]
    assert points.shape[1] == 2 and cells.shape[1] == 4 and cells.max() < len(points)


def test_snapshots_match_receivers(coupled_run):
    # A snapshot point coinciding with nothing special still has to agree
    # with the field: compare the pressure snapshot at the GLL point nearest
    # to the hydrophone with the receiver trace at that time.
    sim, _ = coupled_run
    s = sim.solver
    snap = s.snapshot("acoustic", "pressure")
    pts = s.snapshot_points("acoustic")
    i = np.argmin((pts[0] - 1.0) ** 2 + (pts[1] - 1.0) ** 2)
    assert np.hypot(pts[0, i] - 1.0, pts[1, i] - 1.0) < 1e-12  # (1, 1) is a GLL point here
    trace = s.trace("h", "pressure")[:, 0]
    for k, t in enumerate(s.snapshot_times):
        j = np.argmin(np.abs(s.times - t))
        assert snap[k, 0, i] == pytest.approx(trace[j], abs=1e-6 * np.abs(trace).max())


def test_xdmf(coupled_run, tmp_path):
    _, path = coupled_run
    xdmf = post.write_xdmf(path, tmp_path / "waves.xdmf")
    root = ET.parse(xdmf).getroot()
    steps = root.findall("./Domain/Grid/Grid")
    res = post.read_results(path)
    assert len(steps) == len(res.snapshot_time)
    names = {a.get("Name") for a in steps[-1].iter("Attribute")}
    assert names == {"velocity_x", "velocity_z", "pressure"}
    with h5py.File(tmp_path / "waves.xdmf.h5", "r") as h:
        np.testing.assert_array_equal(h["elastic/velocity_z/3"][()], res.snapshots[("elastic", "velocity")][3, 1])
        assert h["acoustic/cells"].shape == res.snapshot_mesh["acoustic"][1].shape


def test_segy_round_trip(coupled_run, tmp_path):
    _, path = coupled_run
    res = post.read_results(path)
    out = post.write_segy(path, tmp_path / "geo.sgy", "velocity", component="z", source=(2.0, 0.8))
    seg = post.read_segy(out)
    assert seg.data.shape == (4, len(res.time))
    for i in range(4):
        np.testing.assert_array_equal(seg.data[i], res.trace(f"g{i}", "velocity")[:, 1].astype(np.float32))
    np.testing.assert_allclose(seg.group_x, np.linspace(0.5, 3.5, 4), atol=1e-3)
    np.testing.assert_allclose(seg.group_z, -0.5, atol=1e-3)
    np.testing.assert_allclose(seg.source_x, 2.0, atol=1e-3)
    assert seg.dt == pytest.approx(res.time[1] - res.time[0], rel=1e-12)
    assert "insem" in seg.text
    raw = out.read_bytes()
    assert len(raw) == 3600 + 4 * (240 + 4 * len(res.time))
    assert raw[3500] == 2  # revision 2.0


def test_segy_errors(coupled_run, tmp_path):
    _, path = coupled_run
    with pytest.raises(ValueError, match="did not record"):
        post.write_segy(path, tmp_path / "x.sgy", "velocity", receivers=["h"])
    with pytest.raises(ValueError, match="component"):
        post.write_segy(path, tmp_path / "x.sgy", "pressure", component=1)
