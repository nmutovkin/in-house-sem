"""Verification of the 2D Cartesian solver against analytic solutions (docs/DESIGN.md §9).

Tolerances are set for the default single-precision build; the errors
quoted in comments are what the current code achieves.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

import insem  # noqa: E402
from insem import mesh  # noqa: E402

import analytic  # noqa: E402


def open_box(half_width, n, order, material):
    """Square [-w, w]^2 with absorbing boundaries."""
    m = mesh.rectangle((-half_width, half_width), (-half_width, half_width), nx=n, nz=n)
    sim = insem.Simulation(m, insem.Discretization(order=order))
    sim.set_material(0, material)
    for tag in (mesh.BOTTOM, mesh.RIGHT, mesh.TOP, mesh.LEFT):
        sim.set_boundary(tag, "absorbing")
    return sim


def test_acoustic_monopole_matches_green_function():
    c, rho, xr, zr = 1.0, 1.0, 1.5, 0.7
    errors = []
    for order, n in [(4, 32), (6, 32)]:
        sim = open_box(4.0, n, order, insem.Material.acoustic(c, rho))
        stf = insem.ricker(1.0)
        sim.add_source(insem.Source.monopole(0.0, 0.0, stf))
        sim.add_receiver("r", xr, zr, ["pressure"])
        sim.set_time(duration=4.5)
        sim.run()
        s = sim.solver
        p = s.trace("r", "pressure")[:, 0]
        errors.append(analytic.rel_l2(p, analytic.acoustic_pressure(s.times, stf(s.times), xr, zr, c, rho)))
    # About 6 points per minimum wavelength at order 4: 1.3 %; order 6: 0.3 %.
    assert errors[0] < 0.03
    assert errors[1] < 0.6 * errors[0]


def test_acoustic_particle_velocity():
    c, rho, xr, zr = 1.0, 1.0, 1.2, -0.9
    sim = open_box(4.0, 40, 5, insem.Material.acoustic(c, rho))
    stf = insem.ricker(1.0)
    sim.add_source(insem.Source.monopole(0.0, 0.0, stf))
    sim.add_receiver("r", xr, zr, ["particle_velocity"])
    sim.set_time(duration=4.0)
    sim.run()
    s = sim.solver
    v = s.trace("r", "particle_velocity")
    va = analytic.acoustic_particle_velocity(s.times, stf(s.times), xr, zr, c, rho)
    assert analytic.rel_l2(v, va) < 0.01  # 0.4 %


@pytest.mark.parametrize(
    "kind,force,moment",
    [
        ("force", (0.3, 1.0), None),
        ("explosion", None, ((1.0, 0.0), (0.0, 1.0))),
        ("double-couple", None, ((0.0, 1.0), (1.0, 0.0))),
    ],
)
def test_elastic_point_sources_match_green_function(kind, force, moment):
    vp, vs, rho, xr, zr = 2.0, 1.15, 1.0, 1.9, 1.1
    sim = open_box(5.0, 48, 5, insem.Material.elastic(vp, vs, rho))
    stf = insem.ricker(1.0)
    if force is not None:
        sim.add_source(insem.Source.force(0.0, 0.0, force[0], force[1], stf))
    else:
        sim.add_source(insem.Source.moment_tensor(0.0, 0.0, moment[0][0], moment[1][1], moment[0][1], stf))
    sim.add_receiver("r", xr, zr, ["displacement"])
    sim.set_time(duration=4.0)
    sim.run()
    s = sim.solver
    u = s.trace("r", "displacement")
    ua = analytic.elastic_displacement(s.times, stf(s.times), xr, zr, vp, vs, rho, force=force, moment=moment)
    assert analytic.rel_l2(u, ua) < 0.01  # 0.14 %


def test_recorded_derivatives_follow_the_time_integrator():
    # With every-step sampling the traces satisfy the central-difference
    # update exactly (interpolation to the receiver is linear):
    #   u+ = u + dt v + dt^2/2 a,   v+ = v + dt/2 (a + a+)
    sim = open_box(3.0, 24, 4, insem.Material.elastic(2.0, 1.0, 1.0))
    sim.add_source(insem.Source.force(0.0, 0.0, 1.0, 0.0, insem.ricker(1.0)))
    sim.add_receiver("r", 0.8, 0.4, ["displacement", "velocity", "acceleration"])
    sim.set_time(duration=3.0)
    sim.run()
    s = sim.solver
    u, v, a = (s.trace("r", q) for q in ("displacement", "velocity", "acceleration"))
    dt = s.dt
    np.testing.assert_allclose(u[1:], u[:-1] + dt * v[:-1] + 0.5 * dt * dt * a[:-1], atol=1e-5 * np.abs(u).max())
    np.testing.assert_allclose(v[1:], v[:-1] + 0.5 * dt * (a[:-1] + a[1:]), atol=1e-5 * np.abs(v).max())
    assert np.abs(v).max() > 0


def test_spectral_convergence_of_rigid_box_eigenmode():
    # p = cos(pi x) cos(pi z) cos(w t), w = c pi sqrt(2), rigid walls on [0, 1]^2.
    c = 1.0
    w = c * np.pi * np.sqrt(2)
    errors = {}
    for order in (2, 3, 4, 6):
        m = mesh.rectangle((0, 1), (0, 1), nx=3, nz=3)
        sim = insem.Simulation(m, insem.Discretization(order=order))
        sim.set_material(0, insem.Material.acoustic(c, 1.0))
        sim.add_receiver("r", 0.3, 0.6, ["pressure"])
        sim.set_time(duration=4 * np.pi / w, safety=0.1)
        s = sim.solver
        xz = s.coordinates("acoustic")
        p0 = np.cos(np.pi * xz[0]) * np.cos(np.pi * xz[1])
        s.set_state("acoustic", p0[None], np.zeros((1, p0.size)))
        s.run()
        exact = np.cos(np.pi * 0.3) * np.cos(np.pi * 0.6) * np.cos(w * s.times)
        errors[order] = np.abs(s.trace("r", "pressure")[:, 0] - exact).max()
    # Errors: 5e-3, 2e-4, 2e-5, then the single-precision floor (~5e-6).
    assert errors[3] < errors[2] / 10
    assert errors[4] < errors[3] / 5
    assert errors[6] < 2e-5


def test_rayleigh_wave_speed_on_free_surface():
    vp, vs = 2.0, 1.0
    m = mesh.rectangle((-8, 8), (-6, 0), nx=64, nz=24)
    sim = insem.Simulation(m, insem.Discretization(order=4))
    sim.set_material(0, insem.Material.elastic(vp, vs, 1.0))
    for tag in (mesh.BOTTOM, mesh.LEFT, mesh.RIGHT):
        sim.set_boundary(tag, "absorbing")  # top stays traction-free
    sim.add_source(insem.Source.force(0.0, 0.0, 0.0, -1.0, insem.ricker(1.5)))
    sim.add_receiver("near", 2.0, 0.0, ["velocity"])
    sim.add_receiver("far", 5.0, 0.0, ["velocity"])
    sim.set_time(duration=7.5)
    sim.run()
    s = sim.solver
    a = s.trace("near", "velocity")[:, 1]
    b = s.trace("far", "velocity")[:, 1]
    cc = np.correlate(b, a, "full")
    i = int(np.argmax(cc))
    frac = 0.5 * (cc[i - 1] - cc[i + 1]) / (cc[i - 1] - 2 * cc[i] + cc[i + 1])
    lag = (i + frac - (len(a) - 1)) * (s.times[1] - s.times[0])
    measured = 3.0 / lag
    assert measured == pytest.approx(analytic.rayleigh_speed(vp, vs), rel=0.01)  # 0.2 %


def test_stacey_absorbs_normally_incident_waves():
    # A plane wave travelling down a narrow rigid-walled channel meets an
    # absorbing bottom at normal incidence, where Stacey is exact in the
    # continuum: the reflection must be small.
    c = 1.0
    m = mesh.rectangle((0, 0.5), (-6, 0), nx=2, nz=48)
    sim = insem.Simulation(m, insem.Discretization(order=4))
    sim.set_material(0, insem.Material.acoustic(c, 1.0))
    sim.set_boundary(mesh.BOTTOM, "absorbing")
    sim.add_receiver("r", 0.25, -2.0, ["pressure"])
    sim.set_time(duration=10.0)
    s = sim.solver
    xz = s.coordinates("acoustic")
    # Downgoing pulse: p(z, t) = f(z + c t) with v = dp/dt = c f'.
    sigma = 0.3
    f = np.exp(-((xz[1] + 2.0) ** 2) / (2 * sigma**2))
    df = -(xz[1] + 2.0) / sigma**2 * f
    s.set_state("acoustic", f[None], (c * df)[None])
    s.run()
    p = s.trace("r", "pressure")[:, 0]
    t = s.times
    incident = np.abs(p[t < 1.5]).max()
    # The reflection from z = -6 would return to z = -2 after t = 8.
    reflected = np.abs(p[t > 6.5]).max()
    assert reflected < 0.01 * incident


def test_curved_layered_mesh_of_a_homogeneous_medium():
    # Two "layers" of the same material separated by a sinusoidal interface:
    # every element is curved, and the result must still match the
    # homogeneous Green's function.
    vp, vs, rho, xr, zr = 2.0, 1.15, 1.0, 1.7, 1.3
    m = mesh.layered(
        (-5, 5),
        [-5.0, lambda x: 0.6 + 0.4 * np.sin(1.3 * x), 5.0],
        nz=24,
        regions=[0, 1],
        nx=48,
        quadratic=True,
    )
    sim = insem.Simulation(m, insem.Discretization(order=5))
    sim.set_material(0, insem.Material.elastic(vp, vs, rho))
    sim.set_material(1, insem.Material.elastic(vp, vs, rho))
    for tag in (mesh.BOTTOM, mesh.RIGHT, mesh.TOP, mesh.LEFT):
        sim.set_boundary(tag, "absorbing")
    sim.setup()
    assert sim.geometry_kind_counts()["curved"] > 0.9 * sim.dofs.n_elements
    stf = insem.ricker(1.0)
    sim.add_source(insem.Source.force(0.0, 0.0, 0.3, 1.0, stf))
    sim.add_receiver("r", xr, zr, ["displacement"])
    sim.set_time(duration=4.0)
    sim.run()
    s = sim.solver
    ua = analytic.elastic_displacement(s.times, stf(s.times), xr, zr, vp, vs, rho, force=(0.3, 1.0))
    assert analytic.rel_l2(s.trace("r", "displacement"), ua) < 0.01


def test_fluid_solid_normal_incidence_reflection_and_transmission():
    # Downgoing plane pressure pulse in a fluid over a solid. The fluid side
    # walls are rigid (natural), so the wave stays planar; receivers sit on
    # the axis, away from the solid's side edges for the measured window.
    vf, rf, vp, vs, rs = 1.5, 1.0, 3.0, 1.6, 2.2
    m = mesh.layered((-4, 4), [-4.0, 0.0, 4.0], nz=[24, 24], regions=[1, 2], nx=24)
    sim = insem.Simulation(m, insem.Discretization(order=4))
    sim.set_material(1, insem.Material.elastic(vp, vs, rs))
    sim.set_material(2, insem.Material.acoustic(vf, rf))
    sim.set_boundary(mesh.TOP, "absorbing")
    sim.set_boundary(mesh.BOTTOM, "absorbing")
    sim.add_receiver("fluid", 0.0, 1.0, ["pressure"])
    sim.add_receiver("solid", 0.0, -1.0, ["velocity"])
    sim.set_time(duration=2.4)
    s = sim.solver
    xz = s.coordinates("acoustic")
    sigma = 0.15
    f = np.exp(-((xz[1] - 2.0) ** 2) / (2 * sigma**2))
    s.set_state("acoustic", f[None], (vf * -(xz[1] - 2.0) / sigma**2 * f)[None])
    s.run()
    t = s.times
    p = s.trace("fluid", "pressure")[:, 0]
    v = s.trace("solid", "velocity")[:, 1]

    def peak(x):
        return x[np.argmax(np.abs(x))]

    incident = peak(p[t < 1.2])
    zf, zs = rf * vf, rs * vp
    assert peak(p[(t > 1.7) & (t < 2.3)]) / incident == pytest.approx((zs - zf) / (zs + zf), rel=0.01)
    assert peak(v[(t > 1.4) & (t < 2.0)]) / incident == pytest.approx(-2 / (zs + zf), rel=0.01)


def test_scholte_wave_speed():
    # Soft solid (vs < vf) under a fluid: the interface wave travels slower
    # than both the solid's Rayleigh wave and the fluid wave.
    from scipy.optimize import brentq

    vf, rf, vp, vs, rs = 1.5, 1.0, 2.0, 0.8, 1.8
    k = (vs / vp) ** 2

    def scholte(c):
        x = c / vs
        return ((2 - x * x) ** 2 - 4 * np.sqrt(1 - k * x * x) * np.sqrt(1 - x * x)
                + (rf / rs) * x**4 * np.sqrt(1 - k * x * x) / np.sqrt(1 - (c / vf) ** 2))

    expected = brentq(scholte, 0.3 * vs, 0.99999 * vs)
    m = mesh.layered((-8, 8), [-6.0, 0.0, 6.0], nz=[24, 24], regions=[1, 2], nx=64)
    sim = insem.Simulation(m, insem.Discretization(order=4))
    sim.set_material(1, insem.Material.elastic(vp, vs, rs))
    sim.set_material(2, insem.Material.acoustic(vf, rf))
    for tag in (mesh.TOP, mesh.BOTTOM, mesh.LEFT, mesh.RIGHT):
        sim.set_boundary(tag, "absorbing")
    sim.add_source(insem.Source.force(0.0, -0.05, 0.0, 1.0, insem.ricker(1.0)))
    sim.add_receiver("near", 2.0, 0.0, ["velocity"], medium="elastic")
    sim.add_receiver("far", 5.0, 0.0, ["velocity"], medium="elastic")
    sim.set_time(duration=10.0)
    sim.run()
    s = sim.solver
    a = s.trace("near", "velocity")[:, 1]
    b = s.trace("far", "velocity")[:, 1]
    cc = np.correlate(b, a, "full")
    i = int(np.argmax(cc))
    frac = 0.5 * (cc[i - 1] - cc[i + 1]) / (cc[i - 1] - 2 * cc[i] + cc[i + 1])
    lag = (i + frac - (len(a) - 1)) * (s.times[1] - s.times[0])
    assert 3.0 / lag == pytest.approx(expected, rel=0.01)  # 0.16 %


def _channel_trace(sigma, refine, order=4, h=0.5):
    """Downgoing plane pulse in a rigid-walled acoustic channel; the lower half
    is refined one level when `refine` is set (interface at z = -8)."""
    m = mesh.rectangle((0, 2 * h), (-16, 0), nx=2, nz=int(round(16 / h)))
    regions = [(-1, 3, -16, -8, 1)] if refine else []
    sim = insem.Simulation(m, insem.Discretization(order=order, regions=regions))
    sim.set_material(0, insem.Material.acoustic(1.0, 1.0))
    sim.set_boundary(mesh.BOTTOM, "absorbing")
    sim.set_boundary(mesh.TOP, "absorbing")
    sim.add_receiver("r", 0.37 * h, -4.0, ["pressure"])
    sim.set_time(duration=12.0, dt=0.004)
    s = sim.solver
    xz = s.coordinates("acoustic")
    f = np.exp(-((xz[1] + 2.0) ** 2) / (2 * sigma**2))
    s.set_state("acoustic", f[None], (-(xz[1] + 2.0) / sigma**2 * f)[None])
    s.run()
    return s.times, s.trace("r", "pressure")[:, 0]


@pytest.mark.parametrize("sigma,max_db", [(0.3, -60.0), (0.2, -40.0)])
def test_refinement_interface_reflection(sigma, max_db):
    # The difference to an unrefined run isolates the reflection from the
    # coarse-fine (hanging-node) interface. Measured: -66 dB at ~7.5 points
    # per wavelength on the coarse side (sigma = 0.3), -47 dB at ~5 (0.2).
    t, refined = _channel_trace(sigma, True)
    _, uniform = _channel_trace(sigma, False)
    incident = np.abs(uniform[t < 4.0]).max()
    window = (t > 8.0) & (t < 11.5)
    reflection = np.abs(refined[window] - uniform[window]).max() / incident
    assert 20 * np.log10(reflection) < max_db


def test_green_function_on_locally_refined_mesh():
    # Refining a box around the source must not degrade the far-field result.
    c, rho, xr, zr = 1.0, 1.0, 1.5, 0.7
    m = mesh.rectangle((-4, 4), (-4, 4), nx=32, nz=32)
    sim = insem.Simulation(m, insem.Discretization(order=4, regions=[(-0.6, 0.6, -0.6, 0.6, 2)]))
    sim.set_material(0, insem.Material.acoustic(c, rho))
    for tag in (mesh.BOTTOM, mesh.RIGHT, mesh.TOP, mesh.LEFT):
        sim.set_boundary(tag, "absorbing")
    stf = insem.ricker(1.0)
    sim.add_source(insem.Source.monopole(0.05, -0.03, stf))
    sim.add_receiver("r", xr, zr, ["pressure"])
    sim.set_time(duration=4.5)
    sim.run()
    assert sim.dofs.n_elements > 32 * 32
    s = sim.solver
    expected = analytic.acoustic_pressure(s.times, stf(s.times), xr - 0.05, zr + 0.03, c, rho)
    assert analytic.rel_l2(s.trace("r", "pressure")[:, 0], expected) < 0.03
