"""Monopole sonic logging in an open and a cased hole (axisymmetric).

    python examples/borehole/run.py [output_dir] [--no-reference]

Runs a 5 kHz monopole on the axis of a 10 cm borehole in a fast formation,
open hole and cased hole (9 mm steel casing, 4 cm cement), records an
8-receiver array (1.0-3.1 m), compares it with the discrete-wavenumber
reference (insem.reference; skipped with --no-reference), and plots the
mesh, wavefield snapshots, waveforms and slowness-time coherence (STC).
Needs matplotlib and scipy. Takes a few minutes, mostly the cased hole,
whose thin steel annulus sets a small time step.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import insem
from insem import mesh, post
from insem.reference import borehole_axis_pressure

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0]) if args else Path("borehole_out")
out.mkdir(parents=True, exist_ok=True)
with_reference = "--no-reference" not in sys.argv
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

MUD = insem.Material.acoustic(1500.0, 1000.0)
STEEL = insem.Material.elastic(5900.0, 3200.0, 7850.0)
CEMENT = insem.Material.elastic(3000.0, 1700.0, 1900.0)
FORMATION = insem.Material.elastic(4000.0, 2300.0, 2500.0)
F0 = 5000.0
OFFSETS = np.round(np.linspace(1.0, 3.1, 8), 3)
CASES = {
    "open hole": ([("mud", 0.1)], {"mud": MUD, "formation": FORMATION}),
    "cased hole": ([("mud", 0.101), ("steel", 0.110), ("cement", 0.150)],
                   {"mud": MUD, "steel": STEEL, "cement": CEMENT, "formation": FORMATION}),
}
COLORS = {"mud": "#a9d6f5", "steel": "#7d7d7d", "cement": "#c9c0a8", "formation": "#d8c3a5"}


def build(annuli, materials):
    # Tall enough that Stacey reflections of the guided Stoneley wave from the
    # top and bottom (a few % for first-order absorbing boundaries) arrive
    # after the recording window.
    b = mesh.borehole((-2.5, 5.5), annuli, 2.5, dz=0.08, dr=0.3)
    sim = insem.Simulation(b.mesh, insem.Discretization(order=4, f_max=2.5 * F0))
    for name, mat in materials.items():
        sim.set_material(b.regions[name], mat)
    for name in ("bottom", "outer", "top"):
        sim.set_boundary(b.boundaries[name], "absorbing")
    sim.add_source(insem.Source.monopole(0.0, 0.0, insem.ricker(F0)))
    for z in OFFSETS:
        sim.add_receiver(f"z{z}", 0.0, float(z), ["pressure"])
    return b, sim


runs = {}
for name, (annuli, materials) in CASES.items():
    b, sim = build(annuli, materials)
    sim.set_time(duration=2.6e-3, record_every=4, snapshot_every=1)
    dt = sim.solver.dt
    sim.set_time(duration=2.6e-3, record_every=4, snapshot_every=int(round(0.1e-3 / dt)))
    s = sim.solver
    t0 = time.time()
    sim.run()
    print(f"{name}: {sim.dofs.n_elements} elements, dt {s.dt:.2e} s, {s.n_steps} steps, {time.time() - t0:.0f} s")
    sim.write_results(str(out / f"{name.replace(' ', '_')}_results.h5"))
    data = np.array([s.trace(f"z{z}", "pressure")[:, 0] for z in OFFSETS])
    ref = None
    if with_reference:
        t0 = time.time()
        layers = [(materials[n], r) for n, r in annuli]
        ref = borehole_axis_pressure(s.times, insem.ricker(F0)(s.times), OFFSETS, layers,
                                                     FORMATION).T
        err = [np.linalg.norm(a - c) / np.linalg.norm(c) for a, c in zip(data, ref)]
        print(f"  reference {time.time() - t0:.0f} s; SEM vs reference: {100 * min(err):.2f}-{100 * max(err):.2f} %")
    runs[name] = dict(b=b, sim=sim, t=s.times, data=data, ref=ref)

# ------------------------------------------------------------ 1. mesh close-up
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, (name, run) in zip(axes, runs.items()):
    sim, b = run["sim"], run["b"]
    x, z = sim.geometry.x, sim.geometry.z
    loop = lambda a: np.concatenate([a[:, 0, :], a[:, 1:, -1], a[:, -1, -2::-1], a[:, -2::-1, 0]], axis=1)
    polys = np.stack([loop(x), loop(z)], axis=-1)
    region_names = {v: k for k, v in b.regions.items()}
    reg = sim.mesh.material[sim.forest.leaves[:, 0]]
    colors = [COLORS.get(region_names[r], "#d8c3a5") for r in reg]
    axial = np.array([sim.forest.leaves[i, 2] == 0 and sim.mesh.nodes[sim.mesh.cells[sim.forest.leaves[i, 0], 0], 0] == 0
                      for i in range(len(reg))])
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.4))
    ax.add_collection(PolyCollection(polys[axial], facecolors="none", edgecolors="crimson", linewidths=1.0))
    inside = (x.ravel() < 0.45) & (z.ravel() > -0.25) & (z.ravel() < 0.25)
    ax.plot(x.ravel()[inside], z.ravel()[inside], ".", color="k", ms=1)
    ax.plot(0, 0, "*", color="gold", mec="k", ms=14, clip_on=False, zorder=5)
    ax.set_xlim(0, 0.45); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal")
    ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"{name}: {sim.dofs.n_elements} elements, GLL/GLJ points\n(red: axial elements with GLJ points in r)")
    for n, c in COLORS.items():
        if n in b.regions:
            ax.plot([], [], "s", color=c, mec="k", label=n)
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig1_borehole_mesh.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ 2. snapshots
def tris(cells):
    return np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])

fig, axes = plt.subplots(2, 4, figsize=(11, 7.5), sharey=True, layout="constrained")
for row, (name, run) in zip(axes, runs.items()):
    s = run["sim"].solver
    ts = s.snapshot_times
    pa, va = s.snapshot("acoustic", "pressure")[:, 0], s.snapshot("elastic", "velocity")
    pts_a, cells_a = s.snapshot_points("acoustic"), s.snapshot_cells("acoustic")
    pts_e, cells_e = s.snapshot_points("elastic"), s.snapshot_cells("elastic")
    vmag = np.hypot(va[:, 0], va[:, 1])
    for ax, t_want in zip(row, (0.45e-3, 0.8e-3, 1.4e-3, 2.1e-3)):
        k = int(np.argmin(np.abs(ts - t_want)))
        pmax = 0.3 * np.abs(pa[k]).max()
        vmax = 0.5 * vmag[k].max()
        for sign in (1, -1):  # mirror about the axis for display
            ax.tripcolor(sign * pts_e[0], pts_e[1], tris(cells_e), vmag[k], cmap="Greys", vmin=0, vmax=vmax,
                         shading="gouraud")
            ax.tripcolor(sign * pts_a[0], pts_a[1], tris(cells_a), pa[k], cmap="RdBu_r", vmin=-pmax, vmax=pmax,
                         shading="gouraud")
        for rr in [r for _, r in CASES[name][0]]:
            ax.axvline(rr, color="k", lw=0.3); ax.axvline(-rr, color="k", lw=0.3)
        ax.plot(np.zeros_like(OFFSETS), OFFSETS, "v", color="gold", mec="k", ms=4)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.8, 3.6)
        ax.set_title(f"{name}, t = {ts[k] * 1e3:.2f} ms")
        ax.set_xlabel("r (m), mirrored")
    row[0].set_ylabel("z (m)")
fig.suptitle("monopole sonic wavefields: pressure in the mud (red/blue), |velocity| in the solids (grey)")
fig.savefig(out / "fig2_borehole_snapshots.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ 3. waveforms
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (name, run) in zip(axes, runs.items()):
    t, data, ref = run["t"], run["data"], run["ref"]
    scale = 0.45 * (OFFSETS[1] - OFFSETS[0]) / np.abs(data).max()
    for j, z in enumerate(OFFSETS):
        if ref is not None:
            ax.plot(t * 1e3, z + scale * ref[j], color="0.6", lw=2.5, label="reference (DWM)" if j == 0 else None)
        ax.plot(t * 1e3, z + scale * data[j], "k", lw=0.8, label="SEM" if j == 0 else None)
    for slow, label in ((1 / FORMATION.vp, "P"), (1 / FORMATION.vs, "S")):
        ax.plot((1.2 / F0 + slow * OFFSETS) * 1e3, OFFSETS, "--", lw=0.8, color="C3")
        ax.text((1.2 / F0 + slow * OFFSETS[-1]) * 1e3, OFFSETS[-1] + 0.12, label, color="C3")
    ax.set_title(f"{name}: pressure on the axis")
    ax.set_xlabel("time (ms)"); ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("receiver offset (m)")
fig.tight_layout()
fig.savefig(out / "fig3_borehole_waveforms.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ 4. STC
grid = np.linspace(150e-6, 900e-6, 376)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (name, run) in zip(axes, runs.items()):
    t, data = run["t"], run["data"]
    r = post.stc(data, OFFSETS, t[1] - t[0], grid, window=0.2e-3, times=np.arange(0, 2.0e-3, 5e-6))
    im = ax.pcolormesh(r.time * 1e3, r.slowness * 1e6, r.coherence, cmap="magma", vmin=0, vmax=1, shading="auto")
    for band, label in (((0, 350e-6), "P"), ((380e-6, 520e-6), "S"), ((550e-6, 1), "Stoneley")):
        s_pick, t_pick, c = r.peak(s_min=band[0], s_max=band[1])
        ax.plot(t_pick * 1e3, s_pick * 1e6, "o", mfc="none", mec="cyan", ms=10)
        ax.text(t_pick * 1e3 + 0.05, s_pick * 1e6 + 15, f"{label} {s_pick * 1e6:.0f}", color="cyan")
    ax.axhline(1e6 / FORMATION.vp, color="w", lw=0.5, ls="--")
    ax.axhline(1e6 / FORMATION.vs, color="w", lw=0.5, ls="--")
    ax.set_title(f"{name}: slowness-time coherence"); ax.set_xlabel("window start (ms)")
axes[0].set_ylabel("slowness (us/m)")
fig.colorbar(im, ax=axes, label="coherence")
fig.savefig(out / "fig4_borehole_stc.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote figures and results to {out}")
