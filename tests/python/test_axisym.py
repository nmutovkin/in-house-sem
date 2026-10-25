"""Axisymmetric (r, z) solver against 3D analytic solutions (docs/DESIGN.md §2.4)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

import insem  # noqa: E402
from insem import mesh  # noqa: E402

import analytic  # noqa: E402


def axisym_box(material, order, nr=24, nz=48, r1=4.0, half=4.0):
    m = mesh.rectangle((0, r1), (-half, half), nx=nr, nz=nz, coords="axisymmetric")
    sim = insem.Simulation(m, insem.Discretization(order=order))
    sim.set_material(0, material)
    for tag in (mesh.RIGHT, mesh.TOP, mesh.BOTTOM):  # LEFT is the axis
        sim.set_boundary(tag, "absorbing")
    return sim


RECEIVERS = {"axis": (0.0, 1.6), "off": (1.2, -0.9), "near": (0.3, 0.4)}


def test_fluid_point_source_on_axis():
    c, rho = 1.0, 1.0
    errors = {}
    for order in (4, 6):
        sim = axisym_box(insem.Material.acoustic(c, rho), order)
        stf = insem.ricker(1.0)
        sim.add_source(insem.Source.monopole(0.0, 0.0, stf))
        for name, (r, z) in RECEIVERS.items():
            sim.add_receiver(name, r, z, ["pressure"])
        sim.set_time(duration=4.5)
        sim.run()
        s = sim.solver
        for name, (r, z) in RECEIVERS.items():
            exact = analytic.acoustic_pressure_3d(s.times, stf(s.times), r, z, c, rho)
            errors[(order, name)] = analytic.rel_l2(s.trace(name, "pressure")[:, 0], exact)
    # order 4: 0.2-0.7 %; order 6: 0.05-0.17 %
    for name in RECEIVERS:
        assert errors[(4, name)] < 0.015
        assert errors[(6, name)] < 0.5 * errors[(4, name)]


@pytest.mark.parametrize("kind", ["force", "explosion"])
def test_elastic_point_sources_on_axis(kind):
    vp, vs, rho = 2.0, 1.15, 1.0
    sim = axisym_box(insem.Material.elastic(vp, vs, rho), 5, r1=5.0, half=5.0)
    stf = insem.ricker(1.0)
    if kind == "force":
        sim.add_source(insem.Source.force(0.0, 0.0, 0.0, 1.0, stf))
    else:
        sim.add_source(insem.Source.moment_tensor(0.0, 0.0, 1.0, 1.0, 0.0, stf))
    receivers = {"axis": (0.0, 1.8), "off": (1.5, 1.0), "off2": (2.0, -1.2)}
    for name, (r, z) in receivers.items():
        sim.add_receiver(name, r, z, ["displacement"])
    sim.set_time(duration=4.0)
    sim.run()
    s = sim.solver
    for name, (r, z) in receivers.items():
        kwargs = {"force_z": 1.0} if kind == "force" else {"explosion": 1.0}
        exact = analytic.elastic_displacement_3d(s.times, stf(s.times), r, z, vp, vs, rho, **kwargs)
        assert analytic.rel_l2(s.trace(name, "displacement"), exact) < 0.012  # 0.1-0.64 %
